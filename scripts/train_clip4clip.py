import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data.msvd import MSVDFrames, make_collate
from src.eval.clip4clip_eval import build_caption_pool, evaluate_clip4clip
from src.models.clip4clip import contrastive_loss, encode_texts, encode_videos, load_clip4clip
from src.utils.hub import push_checkpoint, try_resume

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max_hours", type=float, default=11.0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev, t0, run = "cuda", time.time(), cfg["run_name"]

ckpt_path, state = try_resume(cfg["hf_repo"], f"{run}/last")
processor, model = load_clip4clip(ckpt_path or cfg["model_name"])
state = state or {"epoch": 0, "best_cider": -1.0, "history": []}
model.to(dev)
if cfg.get("grad_checkpointing", False):
    model.gradient_checkpointing_enable()
    print("gradient checkpointing: ON")
print(f"run={run} starting at epoch {state['epoch']}")

tok = processor.tokenizer
collate = make_collate(tok, cfg["max_words"])
train_ds = MSVDFrames(cfg["data_root"], "train", processor, cfg["num_frames"], True, cfg["repeats"])
val_ds = MSVDFrames(cfg["data_root"], "val", processor, cfg["num_frames"], train=False)
train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
val_dl = DataLoader(val_ds, cfg.get("eval_batch_size", 16), num_workers=4, collate_fn=collate)
pool = build_caption_pool(train_ds.caps)
print(f"train pairs/epoch: {len(train_ds)} | caption pool: {len(pool)}")

opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
total_steps = len(train_dl) * cfg["epochs"]
sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
for _ in range(state["epoch"] * len(train_dl)):
    sched.step()
scaler = torch.amp.GradScaler()

for epoch in range(state["epoch"], cfg["epochs"]):
    model.train()
    running = 0.0
    for b in tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", mininterval=60):
        with torch.autocast("cuda", dtype=torch.float16):
            v = encode_videos(model, b["pixel_values"].to(dev))
            t = encode_texts(model, b["input_ids"].to(dev), b["attention_mask"].to(dev))
        loss = contrastive_loss(v, t, model.logit_scale)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True); sched.step()
        running += loss.item()

    scores, retrieval, preds = evaluate_clip4clip(model, tok, val_ds, val_dl, dev, pool)
    state["epoch"] = epoch + 1
    state["history"].append({"epoch": epoch + 1, "train_loss": round(running / len(train_dl), 4), **scores, **retrieval})
    print(state["history"][-1], "| example:", next(iter(preds.values())))

    if scores["CIDEr"] > state["best_cider"]:
        state["best_cider"] = scores["CIDEr"]
        push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/best")
    push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/last")

    if (time.time() - t0) / 3600 > args.max_hours - 1:
        print("Close to the Kaggle time limit -> stopping. Run the same command again to resume.")
        break
print("training finished:", state["history"][-1])
