"""Idea 3 (step 1): generate K candidate captions per video for one model.

Output: results/candidates/<name>_<split>.json = {video_id: [caption_1, ..., caption_K]}
caption_1 is the model's normal top beam, so the file also reproduces its baseline.

  python scripts/generate_candidates.py --model git   --config configs/git_base_v3.yaml --split test
  python scripts/generate_candidates.py --model blip2 --config configs/blip2.yaml       --split test
  python scripts/generate_candidates.py --model qwen  --config configs/qwen25vl.yaml    --split test
  python scripts/generate_candidates.py --model clip4clip --config configs/clip4clip.yaml --split test
Run each one for --split val as well (the reranker tunes its weight on val).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.msvd import MSVDFrames, make_collate
from src.utils.hub import try_resume


def norm(c):
    return " ".join(c.strip().lower().rstrip(".!").split())


def dedupe(caps, k):
    out = []
    for c in caps:
        c = norm(c)
        if c and c not in out:
            out.append(c)
    return out[:k]


@torch.no_grad()
def run_git(cfg, args):
    from src.models.git_model import load_git
    from src.utils.generate import _cache_frame_features
    path, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert path, "GIT checkpoint not found on HF Hub"
    processor, model = load_git(path, cfg["num_frames"], cfg.get("temporal_adapter"))
    model.to("cuda").eval()
    ds = MSVDFrames(cfg["data_root"], args.split, processor, cfg["num_frames"], train=False)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    dl = DataLoader(ds, args.batch_size or 2, num_workers=4, collate_fn=make_collate(processor.tokenizer))
    out = {}
    for b in tqdm(dl, desc="GIT candidates", mininterval=30):
        restore = _cache_frame_features(model)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                ids = model.generate(pixel_values=b["pixel_values"].cuda(), max_length=20, num_beams=args.k,
                                     num_return_sequences=args.k, use_cache=False)
        finally:
            restore()
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for i, vid in enumerate(b["video_ids"]):
            out[vid] = texts[i * args.k:(i + 1) * args.k]
    return out


@torch.no_grad()
def run_blip2(cfg, args):
    from src.models.blip2_video import load_blip2, make_blip2_collate, try_resume_trainable, video_prefix
    ckpt, _ = try_resume_trainable(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert ckpt, "BLIP-2 checkpoint not found on HF Hub"
    processor, model = load_blip2(cfg["model_name"], cfg["num_frames"], ckpt)
    model.to("cuda").eval()
    tok = processor.tokenizer
    ds = MSVDFrames(cfg["data_root"], args.split, processor.image_processor, cfg["num_frames"], train=False)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    dl = DataLoader(ds, args.batch_size or 4, num_workers=4, collate_fn=make_blip2_collate(tok))
    nl = tok("\n", add_special_tokens=False)["input_ids"][-1]
    out = {}
    for b in tqdm(dl, desc="BLIP-2 candidates", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16):
            prefix = video_prefix(model, b["pixel_values"].cuda())
            lm = model.language_model
            bos = lm.get_input_embeddings()(torch.full((prefix.shape[0], 1), tok.bos_token_id, device="cuda"))
            inputs = torch.cat([prefix.to(bos.dtype), bos], 1)
            mask = torch.ones(inputs.shape[:2], dtype=torch.long, device="cuda")
            ids = lm.generate(inputs_embeds=inputs, attention_mask=mask, num_beams=args.k,
                              num_return_sequences=args.k, max_new_tokens=20,
                              eos_token_id=[nl, tok.eos_token_id], pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        for i, vid in enumerate(b["video_ids"]):
            out[vid] = texts[i * args.k:(i + 1) * args.k]
    return out


@torch.no_grad()
def run_qwen(cfg, args):
    from src.models.qwen25vl import MSVDVideoFrames, clean_caption, load_qwen25vl, make_qwen_collate, model_inputs
    ckpt, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert ckpt, "Qwen LoRA adapter not found on HF Hub"
    processor, model = load_qwen25vl(cfg["model_name"], adapter_path=ckpt, trainable=False)
    model.config.use_cache = True
    model.eval()
    tok = processor.tokenizer
    ds = MSVDVideoFrames(cfg["data_root"], args.split, cfg["num_frames"], cfg["frame_size"], train=False)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    dl = DataLoader(ds, args.batch_size or 1, num_workers=4,
                    collate_fn=make_qwen_collate(processor, cfg["prompt"], cfg["fps"]))
    out = {}
    for b in tqdm(dl, desc="Qwen candidates", mininterval=30):
        inp = model_inputs(b, "cuda")
        inp.pop("labels", None)
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(**inp, max_new_tokens=cfg["max_new_tokens"], num_beams=args.k,
                                 num_return_sequences=args.k, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, vid in enumerate(b["video_ids"]):
            out[vid] = [clean_caption(t) for t in texts[i * args.k:(i + 1) * args.k]]
    return out


@torch.no_grad()
def run_clip4clip(cfg, args):
    """CLIP4Clip's candidates = its top-K retrieved training captions."""
    from src.eval.clip4clip_eval import build_caption_pool
    from src.models.clip4clip import encode_text_list, encode_video_set, load_clip4clip
    path, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert path, "CLIP4Clip checkpoint not found on HF Hub"
    processor, model = load_clip4clip(path)
    model.to("cuda").eval()
    train_ds = MSVDFrames(cfg["data_root"], "train", processor, cfg["num_frames"], train=False)
    ds = MSVDFrames(cfg["data_root"], args.split, processor, cfg["num_frames"], train=False)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    dl = DataLoader(ds, args.batch_size or 16, num_workers=4, collate_fn=make_collate(processor.tokenizer))
    vids, V = encode_video_set(model, dl, "cuda")
    pool = build_caption_pool(train_ds.caps)
    P = encode_text_list(model, processor.tokenizer, pool, "cuda")
    top = (V @ P.t()).topk(args.k * 3, dim=1).indices.tolist()   # extra, because duplicates get removed
    return {v: [pool[i] for i in idx] for v, idx in zip(vids, top)}


RUNNERS = {"git": run_git, "blip2": run_blip2, "qwen": run_qwen, "clip4clip": run_clip4clip}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(RUNNERS))
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=0, help="lower it if the GPU runs out of memory")
    p.add_argument("--limit", type=int, default=0, help="only the first N videos (quick test)")
    p.add_argument("--out_dir", default="results/candidates")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    cands = RUNNERS[args.model](cfg, args)
    cands = {v: dedupe(c, args.k) for v, c in cands.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.model}_{args.split}.json")
    json.dump(cands, open(out, "w"), indent=1)
    n = sum(len(c) for c in cands.values())
    print(f"saved {out}: {len(cands)} videos, {n / max(len(cands), 1):.1f} unique candidates per video")
    print("example:", next(iter(cands.items())))
