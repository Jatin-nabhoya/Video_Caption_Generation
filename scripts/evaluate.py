import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from torch.utils.data import DataLoader
from src.data.msvd import MSVDFrames, make_collate
from src.eval.metrics import score_captions
from src.models.git_model import load_git
from src.utils.generate import generate_captions
from src.utils.hub import try_resume

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", default="best", choices=["zeroshot", "best", "last"])
p.add_argument("--split", default="test")
p.add_argument("--out", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
if args.checkpoint == "zeroshot":
    path = cfg["model_name"]
else:
    path, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/{args.checkpoint}")
    assert path, "checkpoint not found on HF Hub"
processor, model = load_git(path, cfg["num_frames"])
model.to("cuda")
ds = MSVDFrames(cfg["data_root"], args.split, processor, cfg["num_frames"], train=False)
dl = DataLoader(ds, cfg.get("eval_batch_size", 16), num_workers=4, collate_fn=make_collate(processor.tokenizer))
preds = generate_captions(model, processor, dl, "cuda", num_beams=cfg["num_beams"])
scores = score_captions(preds, ds.caps)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
result = {"model": "GIT-base", "run": cfg["run_name"], "checkpoint": args.checkpoint, "split": args.split, "n_videos": len(preds), "scores": scores, "config": cfg}
json.dump(result, open(args.out, "w"), indent=2)
json.dump(preds, open(args.out.replace(".json", "_preds.json"), "w"), indent=2)
print(json.dumps(scores, indent=2))
