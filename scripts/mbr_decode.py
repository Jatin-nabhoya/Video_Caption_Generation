"""Sampled MBR decoding on an already-trained BLIP-2 checkpoint (no training).
The temperature is tuned on VAL, never on test. The beam-search caption competes as one extra hypothesis.

  python scripts/mbr_decode.py --config configs/blip2_msrvtt.yaml \
      --beam_preds results/blip2_msrvtt_v1_best_test_preds.json --out results/mbr_msrvtt_v1_test.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from torch.utils.data import DataLoader

from src.data.msvd import MSVDFrames
from src.eval.mbr import mbr_select, pick_rate, sample_candidates
from src.eval.metrics import score_captions
from src.models.blip2_video import generate_blip2, load_blip2, make_blip2_collate, try_resume_trainable

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["best", "last"])
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=16, help="samples per video")
p.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.7, 1.0], help="tried on val")
p.add_argument("--top_p", type=float, default=1.0)
p.add_argument("--batch", type=int, default=2, help="videos per batch (x n samples each); 2 fits a T4")
p.add_argument("--beam_preds", default=None, help="saved beam-search test captions of this checkpoint")
p.add_argument("--val_limit", type=int, default=0, help="use only the first N val videos for tuning")
p.add_argument("--limit", type=int, default=0, help="test on the first N videos only (smoke test)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
import torch; torch.manual_seed(a.seed)
t0 = time.time()

ckpt, _ = try_resume_trainable(cfg["hf_repo"], f"{cfg['run_name']}/{a.checkpoint}")
assert ckpt, f"checkpoint {cfg['run_name']}/{a.checkpoint} not found on HF Hub"
processor, model = load_blip2(cfg["model_name"], cfg["num_frames"], ckpt)
model.to("cuda").eval()
tok = processor.tokenizer
collate = make_blip2_collate(tok)


def loader(split, limit):
    ds = MSVDFrames(cfg["data_root"], split, processor.image_processor, cfg["num_frames"], train=False)
    if limit:
        ds.ids = ds.ids[:limit]
    return ds, DataLoader(ds, a.batch, num_workers=4, collate_fn=collate)


# 1) tune the temperature on val (beam captions for val are generated here)
val_ds, val_dl = loader("val", a.val_limit)
val_beam = generate_blip2(model, tok, DataLoader(val_ds, 4, num_workers=4, collate_fn=collate), "cuda",
                          num_beams=cfg["num_beams"])
sweep = {"beam": score_captions(val_beam, val_ds.caps)["CIDEr"]}
print(f"val beam search: CIDEr {sweep['beam']:.2f}")
for T in a.temps:
    samples = sample_candidates(model, tok, val_dl, "cuda", n=a.n, temperature=T, top_p=a.top_p)
    chosen, which = mbr_select(samples, val_beam)
    sweep[T] = score_captions(chosen, val_ds.caps)["CIDEr"]
    print(f"val MBR T={T}: CIDEr {sweep[T]:.2f} (beam {sweep['beam']:.2f}) picks {pick_rate(which)}")
best_T = max(a.temps, key=lambda T: sweep[T])
print("chosen temperature:", best_T)

# 2) test
test_ds, test_dl = loader("test", a.limit)
if a.beam_preds and os.path.exists(a.beam_preds):
    beam = json.load(open(a.beam_preds))
    beam = {v: beam[v] for v in test_ds.ids if v in beam}
    print(f"beam captions loaded from {a.beam_preds} ({len(beam)} videos)")
else:
    beam = generate_blip2(model, tok, DataLoader(test_ds, 4, num_workers=4, collate_fn=collate), "cuda",
                          num_beams=cfg["num_beams"])
samples = sample_candidates(model, tok, test_dl, "cuda", n=a.n, temperature=best_T, top_p=a.top_p)
chosen, which = mbr_select(samples, beam)
ids = [v for v in test_ds.ids if v in chosen]
scores = score_captions({v: chosen[v] for v in ids}, test_ds.caps)
beam_scores = score_captions({v: beam[v] for v in ids}, test_ds.caps)
print("test beam:", beam_scores)
print("test MBR :", scores, "| picks", pick_rate(which))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"method": f"sampled MBR (n={a.n}, T={best_T}, CIDEr-D utility) on {cfg['run_name']}/{a.checkpoint}",
           "split": "test", "n_videos": len(ids), "scores": scores, "beam_scores_same_videos": beam_scores,
           "val_sweep": {str(k): v for k, v in sweep.items()}, "temperature": best_T, "picks": pick_rate(which),
           "minutes": round((time.time() - t0) / 60, 1), "config": cfg}, open(a.out, "w"), indent=2)
json.dump(chosen, open(a.out.replace(".json", "_preds.json"), "w"), indent=2)
json.dump(samples, open(a.out.replace(".json", "_samples.json"), "w"))
print("saved", a.out)
