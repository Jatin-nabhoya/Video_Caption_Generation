"""Idea 3 (step 2): pick the best caption from all models' candidates.

Each candidate c for video v gets two scores:
  clip      = cosine(video, caption) from the fine-tuned CLIP4Clip   -> "does it match the video?"
  consensus = CIDEr of c against the OTHER models' candidates for v -> "do the other models agree?"
Both are z-scored within the video, then
    score = alpha * clip + (1 - alpha) * consensus - gamma * beam_rank
gamma keeps each model's own confidence (its top beam = rank 0) in the decision; a large gamma
falls back to the top beams. alpha and gamma are tuned on the validation split, then applied once to test.
CLIP4Clip's candidates vote but are never picked: they were retrieved BECAUSE of their CLIP score,
so the CLIP signal would always favour them.

  python scripts/rerank.py --tune_split val --split test
Needs results/candidates/<model>_<split>.json for both splits (scripts/generate_candidates.py).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from src.data.msvd import MSVDFrames, make_collate
from src.eval.metrics import score_captions
from src.models.clip4clip import encode_text_list, encode_video_set, load_clip4clip
from src.utils.hub import try_resume

NAMES = {"git": "GIT", "blip2": "BLIP-2", "qwen": "Qwen2.5-VL", "clip4clip": "CLIP4Clip"}
ALPHAS = [round(float(a), 1) for a in np.arange(0, 1.01, 0.1)]
GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def tokenize(d):
    """{key: [captions]} -> {key: [tokenized captions]} with the COCO PTB tokenizer (one Java call)."""
    return PTBTokenizer().tokenize({k: [{"caption": c} for c in v] for k, v in d.items()})


def cider_each(gts, res):
    """Per-key CIDEr-D for already tokenized dicts with identical keys."""
    keys = list(gts)
    _, scores = Cider().compute_score({k: gts[k] for k in keys}, {k: res[k] for k in keys})
    return dict(zip(keys, np.asarray(scores, dtype=float)))


def load_candidates(cand_dir, split, models):
    out = {}
    for m in models:
        path = os.path.join(cand_dir, f"{m}_{split}.json")
        if os.path.exists(path):
            out[m] = json.load(open(path))
        else:
            print(f"  [missing] {path} - {m} left out")
    return out


@torch.no_grad()
def clip_scores(c4c_cfg, split, refs, rows, data_root):
    """Cosine similarity between each video and each of its candidates (fine-tuned CLIP4Clip)."""
    path, _ = try_resume(c4c_cfg["hf_repo"], f"{c4c_cfg['run_name']}/best")
    assert path, "CLIP4Clip checkpoint not found on HF Hub"
    processor, model = load_clip4clip(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    ds = MSVDFrames(data_root, split, processor, c4c_cfg["num_frames"], train=False)
    ds.ids = [v for v in ds.ids if v in refs]
    dl = DataLoader(ds, 16, num_workers=4, collate_fn=make_collate(processor.tokenizer))
    vids, V = encode_video_set(model, dl, dev)
    vidx = {v: i for i, v in enumerate(vids)}
    texts = sorted({r["caption"] for r in rows})
    T = encode_text_list(model, processor.tokenizer, texts, dev)
    tidx = {t: i for i, t in enumerate(texts)}
    vi = torch.tensor([vidx[r["video"]] for r in rows], device=dev)
    ti = torch.tensor([tidx[r["caption"]] for r in rows], device=dev)
    return (V[vi] * T[ti]).sum(1).float().cpu().numpy()


def build_rows(split, cands, refs, c4c_cfg, data_root):
    """One row per (video, model, candidate) with clip, consensus and true-CIDEr scores."""
    vids = sorted(set.intersection(*[set(c) for c in cands.values()]) & set(refs))
    rows = [{"video": v, "model": m, "rank": j, "caption": c}
            for v in vids for m in cands for j, c in enumerate(cands[m][v])]
    print(f"[{split}] {len(vids)} videos, {len(rows)} candidates from {list(cands)}")

    # consensus: this candidate vs. every candidate of the OTHER models for the same video
    others, res, gts_true = {}, {}, {}
    for i, r in enumerate(rows):
        key = str(i)
        res[key] = [r["caption"]]
        others[key] = [c for m in cands if m != r["model"] for c in cands[m][r["video"]]] or [r["caption"]]
        gts_true[key] = refs[r["video"]]
    res_t, oth_t, true_t = tokenize(res), tokenize(others), tokenize(gts_true)
    cons = cider_each(oth_t, res_t)
    true = cider_each(true_t, res_t)          # only for the oracle / analysis, never for selection
    clip = clip_scores(c4c_cfg, split, refs, rows, data_root)
    for i, r in enumerate(rows):
        r["consensus"], r["true_cider"], r["clip"] = cons[str(i)], true[str(i)], float(clip[i])

    # z-score both signals inside each video so they are on the same scale
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r["video"], []).append(r)
    for group in by_vid.values():
        for key in ("clip", "consensus"):
            x = np.array([g[key] for g in group])
            z = (x - x.mean()) / (x.std() + 1e-6)
            for g, val in zip(group, z):
                g[f"z_{key}"] = float(val)
    return by_vid


def select(by_vid, alpha, allowed, gamma=0.0):
    preds = {}
    for v, group in by_vid.items():
        pool = [g for g in group if g["model"] in allowed]
        best = max(pool, key=lambda g: alpha * g["z_clip"] + (1 - alpha) * g["z_consensus"] - gamma * g["rank"])
        preds[v] = best["caption"]
    return preds


def fast_cider(preds, refs):
    t = tokenize({v: refs[v] for v in preds}); r = tokenize({v: [c] for v, c in preds.items()})
    return float(np.mean(list(cider_each(t, r).values()))) * 100


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/kaggle/input/datasets/jatinnabhoya/video-caption-generation/msvd")
    p.add_argument("--clip4clip_config", default="configs/clip4clip.yaml")
    p.add_argument("--cand_dir", default="results/candidates")
    p.add_argument("--models", nargs="+", default=["git", "blip2", "qwen", "clip4clip"])
    p.add_argument("--select_from", nargs="+", default=None,
                   help="models whose captions may be chosen (default: all except clip4clip); the rest still vote")
    p.add_argument("--tune_split", default="val")
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", default="results/rerank")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ann = json.load(open(os.path.join(a.data_root, "msvd_annotations.json")))
    c4c_cfg = yaml.safe_load(open(a.clip4clip_config))
    allowed = set(a.select_from or [m for m in a.models if m != "clip4clip"])

    # 1) tune alpha on validation
    val = build_rows(a.tune_split, load_candidates(a.cand_dir, a.tune_split, a.models), ann[a.tune_split],
                     c4c_cfg, a.data_root)
    val_refs = ann[a.tune_split]
    grid = {(al, ga): fast_cider(select(val, al, allowed, ga), val_refs) for al in ALPHAS for ga in GAMMAS}
    alpha, gamma = max(grid, key=grid.get)
    print(f"validation: best alpha={alpha} gamma={gamma} -> CIDEr {grid[(alpha, gamma)]:.2f}")
    for ga in GAMMAS:
        print(f"  gamma={ga:<4}", " ".join(f"{grid[(al, ga)]:6.1f}" for al in ALPHAS), "  (alpha 0.0 ... 1.0)")

    # 2) apply once to test
    cands = load_candidates(a.cand_dir, a.split, a.models)
    test = build_rows(a.split, cands, ann[a.split], c4c_cfg, a.data_root)
    refs = {v: ann[a.split][v] for v in test}
    systems = {f"{NAMES[m]} (top beam)": {v: cands[m][v][0] for v in test} for m in cands}
    systems["Rerank: consensus only"] = select(test, 0.0, allowed)
    systems["Rerank: CLIP only"] = select(test, 1.0, allowed)
    best_name = f"Rerank: tuned on val (alpha={alpha}, gamma={gamma})"
    systems[best_name] = select(test, alpha, allowed, gamma)
    systems["Oracle (best candidate, upper bound)"] = {
        v: max((g for g in grp if g["model"] in allowed), key=lambda g: g["true_cider"])["caption"]
        for v, grp in test.items()}

    table = {name: score_captions(pr, refs) for name, pr in systems.items()}
    final = systems[best_name]
    chosen_from = {}
    for v, grp in test.items():
        for g in grp:
            if g["caption"] == final[v]:
                chosen_from[NAMES[g["model"]]] = chosen_from.get(NAMES[g["model"]], 0) + 1
                break

    json.dump(final, open(os.path.join(a.out_dir, f"rerank_{a.split}_preds.json"), "w"), indent=2)
    json.dump({"alpha": alpha, "gamma": gamma, "val_grid": {f"{k[0]},{k[1]}": v for k, v in grid.items()}, "scores": table, "chosen_from": chosen_from,
               "models": a.models, "select_from": sorted(allowed)},
              open(os.path.join(a.out_dir, f"rerank_{a.split}_summary.json"), "w"), indent=2)
    json.dump([g for grp in test.values() for g in grp],
              open(os.path.join(a.out_dir, f"rerank_{a.split}_candidates_scored.json"), "w"))

    print(f"\n{'System':45s} BLEU-4  METEOR  ROUGE-L   CIDEr")
    for name, s in table.items():
        print(f"{name:45s} {s['BLEU-4']:6.2f} {s['METEOR']:7.2f} {s['ROUGE-L']:8.2f} {s['CIDEr']:7.2f}")
    print("\nfinal caption came from:", chosen_from)
