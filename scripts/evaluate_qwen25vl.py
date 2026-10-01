import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from torch.utils.data import DataLoader

from src.eval.metrics import score_captions
from src.utils.hub import try_resume
from src.models.qwen25vl import MSVDVideoFrames, generate_qwen, load_qwen25vl, make_qwen_collate

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["zeroshot", "best", "last"])
p.add_argument("--split", default="test")
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--num_beams", type=int, default=None, help="override config (use 1 if beam search runs out of memory)")
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))

ckpt = None
if args.checkpoint != "zeroshot":
    ckpt, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert ckpt, "checkpoint not found on HF Hub"
processor, model = load_qwen25vl(cfg["model_name"], adapter_path=ckpt, trainable=False)
model.config.use_cache = True
model.eval()

ds = MSVDVideoFrames(cfg["data_root"], args.split, cfg["num_frames"], cfg["frame_size"], train=False)
if args.limit:
    ds.ids = ds.ids[:args.limit]
dl = DataLoader(ds, cfg["eval_batch_size"], num_workers=4, collate_fn=make_qwen_collate(processor, cfg["prompt"], cfg["fps"]))
beams = args.num_beams or cfg["num_beams"]
preds, raw = generate_qwen(model, processor, dl, "cuda", num_beams=beams, max_new_tokens=cfg["max_new_tokens"])
scores = score_captions(preds, ds.caps)
avg_len = sum(len(c.split()) for c in preds.values()) / len(preds)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
result = {"model": "Qwen2.5-VL-3B-Instruct (QLoRA)", "run": cfg["run_name"], "checkpoint": args.checkpoint,
          "split": args.split, "n_videos": len(preds), "num_beams": beams, "avg_caption_words": round(avg_len, 2),
          "scores": scores, "config": cfg}
json.dump(result, open(args.out, "w"), indent=2)
json.dump(preds, open(args.out.replace(".json", "_preds.json"), "w"), indent=2)
json.dump(raw, open(args.out.replace(".json", "_raw.json"), "w"), indent=2)   # un-cleaned model output
print(json.dumps(scores, indent=2), "| avg words:", round(avg_len, 2))
print("examples:", list(preds.items())[:3])
