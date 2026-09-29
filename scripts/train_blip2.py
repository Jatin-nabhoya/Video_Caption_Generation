import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data.msvd import MSVDFrames
from src.eval.metrics import score_captions
from src.models.git_model import caption_loss
from src.models.blip2_video import (generate_blip2, lm_logits, load_blip2, make_blip2_collate,
                                    push_trainable, try_resume_trainable, video_prefix)

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max_hours", type=float, default=11.0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev, t0, run = "cuda", time.time(), cfg["run_name"]

# ---- resume automatically (only the trained parts are stored on HF)
ckpt, state = try_resume_trainable(cfg["hf_repo"], f"{run}/last")
processor, model = load_blip2(cfg["model_name"], cfg["num_frames"], ckpt)
state = state or {"epoch": 0, "best_cider": -1.0, "history": []}
model.to(dev)
model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
tok = processor.tokenizer
print(f"run={run} starting at epoch {state['epoch']}")

collate = make_blip2_collate(tok, cfg["max_len"])
train_ds = MSVDFrames(cfg["data_root"], "train", processor.image_processor, cfg["num_frames"], True, cfg["repeats"])
val_ds = MSVDFrames(cfg["data_root"], "val", processor.image_processor, cfg["num_frames"], train=False)
train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
val_dl = DataLoader(val_ds, cfg.get("eval_batch_size", 4), num_workers=4, collate_fn=collate)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
steps_per_epoch = len(train_dl) // cfg["grad_accum"]
total_steps = steps_per_epoch * cfg["epochs"]
sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
for _ in range(state["epoch"] * steps_per_epoch):
    sched.step()
scaler = torch.amp.GradScaler()

for epoch in range(state["epoch"], cfg["epochs"]):
    model.train(); model.vision_model.eval()
    running = 0.0
    for i, b in enumerate(tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", mininterval=60)):
        with torch.autocast("cuda", dtype=torch.float16):
            prefix = video_prefix(model, b["pixel_values"].to(dev))
            logits = lm_logits(model, prefix, b["input_ids"].to(dev), b["attention_mask"].to(dev))
        loss = caption_loss(logits, b["labels"].to(dev))
        scaler.scale(loss / cfg["grad_accum"]).backward()
        running += loss.item()
        if (i + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()

    preds = generate_blip2(model, tok, val_dl, dev, num_beams=1)
    scores = score_captions(preds, val_ds.caps)
    state["epoch"] = epoch + 1
    state["history"].append({"epoch": epoch + 1, "train_loss": round(running / len(train_dl), 4), **scores})
    print(state["history"][-1], "| example:", next(iter(preds.values())))

    if scores["CIDEr"] > state["best_cider"]:
        state["best_cider"] = scores["CIDEr"]
        push_trainable(model, state, cfg["hf_repo"], f"{run}/best")
    push_trainable(model, state, cfg["hf_repo"], f"{run}/last")

    if (time.time() - t0) / 3600 > args.max_hours - 1:
        print("Close to the Kaggle time limit -> stopping. Run the same command again to resume.")
        break
print("training finished:", state["history"][-1])
