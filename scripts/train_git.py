import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from src.data.msvd import MSVDFrames, make_collate
from src.eval.metrics import score_captions
from src.models.git_model import load_git
from src.utils.generate import generate_captions
from src.utils.hub import push_checkpoint, try_resume

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max_hours", type=float, default=11.0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev, t0, run = "cuda", time.time(), cfg["run_name"]
ckpt_path, state = try_resume(cfg["hf_repo"], f"{run}/last")
processor, model = load_git(ckpt_path or cfg["model_name"], cfg["num_frames"])
state = state or {"epoch": 0, "best_cider": -1.0, "history": []}
model.to(dev)
print(f"run={run} starting at epoch {state['epoch']}")

collate = make_collate(processor.tokenizer, cfg["max_len"])
train_ds = MSVDFrames(cfg["data_root"], "train", processor, cfg["num_frames"], True, cfg["repeats"])
val_ds = MSVDFrames(cfg["data_root"], "val", processor, cfg["num_frames"], train=False)
train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
val_dl = DataLoader(val_ds, cfg["batch_size"], num_workers=4, collate_fn=collate)

opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
steps_per_epoch = len(train_dl) // cfg["grad_accum"]
total_steps = steps_per_epoch * cfg["epochs"]
sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
for _ in range(state["epoch"] * steps_per_epoch):
    sched.step()
scaler = torch.amp.GradScaler()

for epoch in range(state["epoch"], cfg["epochs"]):
    model.train(); running = 0.0
    for i, b in enumerate(tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", mininterval=60)):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=b["pixel_values"].to(dev), input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), labels=b["labels"].to(dev))
        scaler.scale(out.loss / cfg["grad_accum"]).backward(); running += out.loss.item()
        if (i + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

    preds = generate_captions(model, processor, val_dl, dev, num_beams=1)
    scores = score_captions(preds, val_ds.caps)
    state["epoch"] = epoch + 1
    state["history"].append({"epoch": epoch + 1, "train_loss": round(running / len(train_dl), 4), **scores})
    print(state["history"][-1], "| example:", next(iter(preds.values())))
    if scores["CIDEr"] > state["best_cider"]:
        state["best_cider"] = scores["CIDEr"]
        push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/best")
    push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/last")
    if (time.time() - t0) / 3600 > args.max_hours - 1:
        print("Close to the Kaggle time limit -> stopping. Run the same command again to resume.")
        break
print("training finished:", state["history"][-1])
