import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from torch.utils.data import DataLoader

from src.data.msvd import MSVDFrames
from src.eval.metrics import score_captions
from src.models.blip2_video import generate_blip2, load_blip2, make_blip2_collate, try_resume_trainable

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["zeroshot", "best", "last"])
p.add_argument("--split", default="test")
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))

ckpt = None
if args.checkpoint != "zeroshot":
    ckpt, _ = try_resume_trainable(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert ckpt, "checkpoint not found on HF Hub"
processor, model = load_blip2(cfg["model_name"], cfg["num_frames"], ckpt)
model.to("cuda").eval()

ds = MSVDFrames(cfg["data_root"], args.split, processor.image_processor, cfg["num_frames"], train=False)
if args.limit:
    ds.ids = ds.ids[:args.limit]
dl = DataLoader(ds, cfg.get("eval_batch_size", 4), num_workers=4, collate_fn=make_blip2_collate(processor.tokenizer))
preds = generate_blip2(model, processor.tokenizer, dl, "cuda", num_beams=cfg["num_beams"])
scores = score_captions(preds, ds.caps)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
result = {"model": "BLIP-2 (OPT-2.7B, Q-Former fine-tuned)", "run": cfg["run_name"], "checkpoint": args.checkpoint,
          "split": args.split, "n_videos": len(preds), "scores": scores, "config": cfg}
json.dump(result, open(args.out, "w"), indent=2)
json.dump(preds, open(args.out.replace(".json", "_preds.json"), "w"), indent=2)
print(json.dumps(scores, indent=2))
print("examples:", list(preds.items())[:3])
