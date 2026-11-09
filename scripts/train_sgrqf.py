"""Train SG-RQF (BLIP-2 + similarity-gated retrieval Q-Former). Same recipe as train_blip2.py,
so the only difference to the BLIP-2 baseline is the method itself.
  python scripts/train_sgrqf.py --config configs/sgrqf_msvd.yaml
Resumes automatically from <hf_repo>/<run_name>/last."""
import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.eval.metrics import score_captions
from src.models.blip2_video import lm_logits, push_trainable, try_resume_trainable
from src.models.git_model import caption_loss
from src.models.sgrqf import (SGRQF_PREFIXES, SGRQFFrames, generate_sgrqf, load_sgrqf, make_sgrqf_collate,
                              similarity_edges, video_prefix_sgrqf)

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max_hours", type=float, default=11.0)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
rc = cfg["sgrqf"]
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev, t0, run = "cuda", time.time(), cfg["run_name"]

ckpt, state = try_resume_trainable(cfg["hf_repo"], f"{run}/last")
init = None
if ckpt is None and rc.get("init_from"):            # v2: warm start from the trained BLIP-2 baseline
    init, _ = try_resume_trainable(cfg["hf_repo"], rc["init_from"])
    assert init, f"baseline checkpoint {rc['init_from']} not found on the HF Hub"
processor, bert_tok, model = load_sgrqf(cfg["model_name"], cfg["num_frames"], ckpt, init_path=init)
edges = similarity_edges(rc["hints_dir"], rc["k"])
state = state or {"epoch": 0, "best_cider": -1.0, "history": [], "similarity_edges": edges}
model.to(dev)
model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
tok = processor.tokenizer
print(f"run={run} starting at epoch {state['epoch']} | reliability bucket edges {edges}")

collate = make_sgrqf_collate(tok, bert_tok, edges, cfg["max_len"])
common = dict(hints_dir=rc["hints_dir"], k=rc["k"], select=rc["select"])
train_ds = SGRQFFrames(cfg["data_root"], "train", processor.image_processor, cfg["num_frames"], True,
                       cfg["repeats"], dropout=rc["dropout"], **common)
val_ds = SGRQFFrames(cfg["data_root"], "val", processor.image_processor, cfg["num_frames"], False, **common)
train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
val_dl = DataLoader(val_ds, cfg.get("eval_batch_size", 4), num_workers=4, collate_fn=collate)
ex = train_ds[0]
print("example hints:", ex["video_id"], ex["hints"])

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
            prefix = video_prefix_sgrqf(model, b["pixel_values"].to(dev), b["hint_ids"].to(dev),
                                        b["hint_mask"].to(dev), b["hint_rel"].to(dev))
            logits = lm_logits(model, prefix, b["input_ids"].to(dev), b["attention_mask"].to(dev))
        loss = caption_loss(logits, b["labels"].to(dev))
        scaler.scale(loss / cfg["grad_accum"]).backward()
        running += loss.item()
        if (i + 1) % cfg["grad_accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()

    preds = generate_sgrqf(model, tok, val_dl, dev, num_beams=1)
    scores = score_captions(preds, val_ds.caps)
    state["epoch"] = epoch + 1
    rel = model.hint_reliability.weight.detach().norm(dim=1).tolist()
    state["history"].append({"epoch": epoch + 1, "train_loss": round(running / len(train_dl), 4), **scores,
                             "reliability_norms": [round(x, 3) for x in rel]})
    print(state["history"][-1], "| example:", next(iter(preds.values())))

    if scores["CIDEr"] > state["best_cider"]:
        state["best_cider"] = scores["CIDEr"]
        push_trainable(model, state, cfg["hf_repo"], f"{run}/best", prefixes=SGRQF_PREFIXES)
    push_trainable(model, state, cfg["hf_repo"], f"{run}/last", prefixes=SGRQF_PREFIXES)

    if (time.time() - t0) / 3600 > args.max_hours - 1:
        print("Close to the Kaggle time limit -> stopping. Run the same command again to resume.")
        break
print("training finished:", state["history"][-1])
