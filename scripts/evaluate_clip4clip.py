import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from torch.utils.data import DataLoader

from src.data.msvd import MSVDFrames, make_collate
from src.eval.clip4clip_eval import build_caption_pool, evaluate_clip4clip
from src.models.clip4clip import load_clip4clip
from src.utils.hub import try_resume

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["zeroshot", "best", "last"])
p.add_argument("--split", default="test")
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))

if args.checkpoint == "zeroshot":
    path = cfg["model_name"]
else:
    path, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert path, "checkpoint not found on HF Hub"
processor, model = load_clip4clip(path)
model.to("cuda").eval()

train_ds = MSVDFrames(cfg["data_root"], "train", processor, cfg["num_frames"], train=False)
ds = MSVDFrames(cfg["data_root"], args.split, processor, cfg["num_frames"], train=False)
if args.limit:
    ds.ids = ds.ids[:args.limit]
dl = DataLoader(ds, cfg.get("eval_batch_size", 16), num_workers=4, collate_fn=make_collate(processor.tokenizer))
scores, retrieval, preds = evaluate_clip4clip(model, processor.tokenizer, ds, dl, "cuda",
                                              build_caption_pool(train_ds.caps))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
result = {"model": "CLIP4Clip (meanP, retrieval captioning)", "run": cfg["run_name"],
          "checkpoint": args.checkpoint, "split": args.split, "n_videos": len(preds),
          "scores": scores, "retrieval": retrieval, "config": cfg}
json.dump(result, open(args.out, "w"), indent=2)
json.dump(preds, open(args.out.replace(".json", "_preds.json"), "w"), indent=2)
print(json.dumps({"captioning": scores, "retrieval (text->video)": retrieval}, indent=2))
