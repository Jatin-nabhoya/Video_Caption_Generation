import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.eval.metrics import score_captions
from src.utils.hub import push_checkpoint, try_resume
from src.models.qwen25vl import (MSVDVideoFrames, generate_qwen, load_qwen25vl, make_qwen_collate, model_inputs)

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max_hours", type=float, default=11.0)
p.add_argument("--limit", type=int, default=0, help="smoke test: use only N train/val videos")
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev, t0, run = "cuda", time.time(), cfg["run_name"]
if args.limit:
    run += "_smoke"                 # smoke tests never touch the real run's checkpoints
    cfg["epochs"] = 1

# ---- resume automatically (only the LoRA adapter is stored on HF)
ckpt, state = try_resume(cfg["hf_repo"], f"{run}/last")
processor, model = load_qwen25vl(cfg["model_name"], adapter_path=ckpt, lora_cfg=cfg["lora"], trainable=True)
state = state or {"epoch": 0, "best_cider": -1.0, "history": []}
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
print(f"run={run} starting at epoch {state['epoch']}")

collate = make_qwen_collate(processor, cfg["prompt"], cfg["fps"])
train_ds = MSVDVideoFrames(cfg["data_root"], "train", cfg["num_frames"], cfg["frame_size"], True, cfg["repeats"])
val_ds = MSVDVideoFrames(cfg["data_root"], "val", cfg["num_frames"], cfg["frame_size"], train=False)
if args.limit:
    train_ds.ids, val_ds.ids = train_ds.ids[:args.limit], val_ds.ids[:args.limit]
train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
val_dl = DataLoader(val_ds, cfg["eval_batch_size"], num_workers=4, collate_fn=collate)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
steps_per_epoch = max(1, len(train_dl) // cfg["grad_accum"])
total_steps = steps_per_epoch * cfg["epochs"]
sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
for _ in range(state["epoch"] * steps_per_epoch):
    sched.step()
scaler = torch.amp.GradScaler()

for epoch in range(state["epoch"], cfg["epochs"]):
    model.train()
    running, n_bad = 0.0, 0
    for i, b in enumerate(tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", mininterval=60)):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**model_inputs(b, dev)).loss
        if not torch.isfinite(loss):          # fp16 overflow guard: skip the batch instead of poisoning the weights
            n_bad += 1; opt.zero_grad(set_to_none=True); continue
        scaler.scale(loss / cfg["grad_accum"]).backward()
        running += loss.item()
        if (i + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()
        if i % 200 == 0:
            print(f"step {i} loss {loss.item():.4f}", flush=True)

    model.config.use_cache = True
    preds, _ = generate_qwen(model, processor, val_dl, dev, num_beams=1, max_new_tokens=cfg["max_new_tokens"])
    model.config.use_cache = False
    scores = score_captions(preds, val_ds.caps)
    state["epoch"] = epoch + 1
    state["history"].append({"epoch": epoch + 1, "train_loss": round(running / max(1, len(train_dl) - n_bad), 4),
                             "skipped_nan_batches": n_bad, **scores})
    print(state["history"][-1], "| example:", next(iter(preds.values())))

    if scores["CIDEr"] > state["best_cider"]:
        state["best_cider"] = scores["CIDEr"]
        push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/best")
    push_checkpoint(model, processor, state, cfg["hf_repo"], f"{run}/last")

    if (time.time() - t0) / 3600 > args.max_hours - 1:
        print("Close to the Kaggle time limit -> stopping. Run the same command again to resume.")
        break
print("training finished:", state["history"][-1])
