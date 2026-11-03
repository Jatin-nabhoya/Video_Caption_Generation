"""Evaluate SG-RQF. With --tune_gate the similarity threshold is chosen on VAL (never on test):
hints whose retrieval similarity is below the q-quantile of the train similarities are masked out.
  python scripts/evaluate_sgrqf.py --config configs/sgrqf_msvd.yaml --tune_gate --out results/sgrqf_msvd_test.json
  python scripts/evaluate_sgrqf.py --config ... --no_hints   # ablation: same model, hints removed"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.eval.metrics import score_captions
from src.models.blip2_video import try_resume_trainable
from src.models.sgrqf import (SGRQFFrames, generate_sgrqf, load_hint_meta, load_sgrqf, make_sgrqf_collate,
                              similarity_edges)

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["best", "last"])
p.add_argument("--split", default="test")
p.add_argument("--out", required=True)
p.add_argument("--tune_gate", action="store_true", help="pick the gate quantile on val")
p.add_argument("--gate_quantiles", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
p.add_argument("--gate_quantile", type=float, default=0.0, help="fixed gate (used when not tuning)")
p.add_argument("--no_hints", action="store_true")
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
rc = cfg["sgrqf"]

ckpt, state = try_resume_trainable(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
assert ckpt, "checkpoint not found on HF Hub"
processor, bert_tok, model = load_sgrqf(cfg["model_name"], cfg["num_frames"], ckpt)
model.to("cuda").eval()
tok = processor.tokenizer
edges = state.get("similarity_edges") or similarity_edges(rc["hints_dir"], rc["k"])
collate = make_sgrqf_collate(tok, bert_tok, edges, cfg["max_len"])
train_sims = [s for rows in load_hint_meta(rc["hints_dir"], "train", rc["k"]).values() for _, s in rows]


def run(split, q, beams, limit=0):
    gate = float(np.quantile(train_sims, q)) if q > 0 else None
    ds = SGRQFFrames(cfg["data_root"], split, processor.image_processor, cfg["num_frames"], False,
                     hints_dir=rc["hints_dir"], k=0 if args.no_hints else rc["k"], select=rc["select"], gate=gate)
    if limit:
        ds.ids = ds.ids[:limit]
    dl = DataLoader(ds, cfg.get("eval_batch_size", 4), num_workers=4, collate_fn=collate)
    preds = generate_sgrqf(model, tok, dl, "cuda", num_beams=beams)
    return preds, score_captions(preds, ds.caps), gate


q_best, sweep = args.gate_quantile, {}
if args.tune_gate and not args.no_hints:
    for q in args.gate_quantiles:
        _, s, gate = run("val", q, beams=1)
        sweep[q] = {"gate": gate, **s}
        print(f"val gate quantile {q:.1f} (sim >= {gate}) -> CIDEr {s['CIDEr']:.2f}")
    q_best = max(sweep, key=lambda q: sweep[q]["CIDEr"])
    print("chosen gate quantile:", q_best)

preds, scores, gate = run(args.split, q_best, beams=cfg["num_beams"], limit=args.limit)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
result = {"model": "SG-RQF (BLIP-2 OPT-2.7B + similarity-gated retrieval Q-Former)", "run": cfg["run_name"],
          "checkpoint": args.checkpoint, "split": args.split, "n_videos": len(preds), "scores": scores,
          "no_hints": args.no_hints, "gate_quantile": q_best, "gate_similarity": gate, "val_gate_sweep": sweep,
          "config": cfg}
json.dump(result, open(args.out, "w"), indent=2)
json.dump(preds, open(args.out.replace(".json", "_preds.json"), "w"), indent=2)
print(json.dumps(scores, indent=2))
print("examples:", list(preds.items())[:3])
