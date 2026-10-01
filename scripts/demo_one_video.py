"""Live demo: one MSVD test video -> CLIP4Clip, GIT and BLIP-2 (zero-shot AND fine-tuned).
Saves  results/demo/<video_id>.png  (frames + reference captions + every model's caption)
and    results/demo/<video_id>.gif  (the clip as an animation, for slides).
Usage: python scripts/demo_one_video.py [--video VIDEO_ID] [--seed 7] [--skip blip2]"""
import argparse, gc, glob, json, os, random, sys, textwrap, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.data.msvd import MSVDFrames

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COLOR = {"CLIP4Clip": "#2a78d6", "GIT": "#eb6834", "BLIP-2": "#1baf7a"}
TEXT, TEXT2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def cfg(name):
    return yaml.safe_load(open(f"configs/{name}.yaml"))


def pixels(processor, root, vid, num_frames):
    ds = MSVDFrames(root, "test", processor, num_frames, train=False)
    return ds[ds.ids.index(vid)]["pixel_values"].unsqueeze(0).to(DEV)


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- the three models
def run_git(vid, root):
    from src.models.git_model import load_git
    from src.utils.generate import _generate
    from src.utils.hub import try_resume
    c = cfg("git_base"); out = {}
    for setting in ["zero-shot", "fine-tuned"]:
        path = c["model_name"] if setting == "zero-shot" else try_resume(c["hf_repo"], f"{c['run_name']}/best")[0]
        proc, model = load_git(path, c["num_frames"]); model.to(DEV).eval()
        with torch.no_grad():
            ids = _generate(model, pixels(proc, root, vid, c["num_frames"]), c.get("num_beams", 3), 20)
        out[setting] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        free(model)
    return out


def run_clip4clip(vid, root, train_caps):
    from src.eval.clip4clip_eval import build_caption_pool
    from src.models.clip4clip import encode_text_list, encode_videos, load_clip4clip
    from src.utils.hub import try_resume
    c = cfg("clip4clip"); pool = build_caption_pool(train_caps); out = {}
    for setting in ["zero-shot", "fine-tuned"]:
        path = c["model_name"] if setting == "zero-shot" else try_resume(c["hf_repo"], f"{c['run_name']}/best")[0]
        proc, model = load_clip4clip(path); model.to(DEV).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            v = encode_videos(model, pixels(proc, root, vid, c["num_frames"]))
        P = encode_text_list(model, proc.tokenizer, pool, DEV)
        sim, idx = (v.float() @ P.float().t()).squeeze(0).topk(3)
        out[setting] = pool[idx[0]]
        out[setting + " top-3"] = [(pool[i], round(float(s), 3)) for s, i in zip(sim, idx)]
        free(model, P)
    return out


def run_blip2(vid, root):
    from src.models.blip2_video import generate_blip2, load_blip2, try_resume_trainable
    c = cfg("blip2"); out = {}
    proc, model = load_blip2(c["model_name"], c["num_frames"]); model.to(DEV).eval()
    batch = [{"pixel_values": pixels(proc.image_processor, root, vid, c["num_frames"]).cpu(), "video_ids": [vid]}]
    out["zero-shot"] = generate_blip2(model, proc.tokenizer, batch, DEV, num_beams=c["num_beams"])[vid]
    ckpt, _ = try_resume_trainable(c["hf_repo"], f"{c['run_name']}/best")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)  # swap in the fine-tuned Q-Former
    out["fine-tuned"] = generate_blip2(model, proc.tokenizer, batch, DEV, num_beams=c["num_beams"])[vid]
    free(model)
    return out


# ---------------------------------------------------------------- output
def make_gif(root, vid, path):
    frames = [Image.open(f).convert("RGB") for f in sorted(glob.glob(f"{root}/frames/{vid}/*.jpg"))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0)


def make_figure(root, vid, refs, results, path):
    files = sorted(glob.glob(f"{root}/frames/{vid}/*.jpg"))
    show = [files[round(i * (len(files) - 1) / 5)] for i in range(6)]
    fig = plt.figure(figsize=(13, 7.2), facecolor=SURFACE)
    for i, f in enumerate(show):
        ax = fig.add_axes([0.02 + i * 0.163, 0.62, 0.155, 0.3]); ax.imshow(Image.open(f)); ax.axis("off")
        ax.set_title(f"frame {i + 1}", fontsize=9, color=TEXT2)
    fig.text(0.02, 0.965, f"Input: MSVD test video  {vid}", fontsize=13, fontweight="bold", color=TEXT)
    y = 0.55
    fig.text(0.02, y, "Human reference captions (3 of %d):" % len(refs), fontsize=10.5, fontweight="bold", color=TEXT)
    for r in refs[:3]:
        y -= 0.04; fig.text(0.04, y, f"• {r}", fontsize=10, color=TEXT2)
    y -= 0.07
    fig.text(0.02, y, "Model", fontsize=10.5, fontweight="bold", color=TEXT)
    fig.text(0.17, y, "Zero-shot (no training on MSVD)", fontsize=10.5, fontweight="bold", color=TEXT)
    fig.text(0.58, y, "Fine-tuned on MSVD (ours)", fontsize=10.5, fontweight="bold", color=TEXT)
    for m in ["CLIP4Clip", "GIT", "BLIP-2"]:
        if m not in results:
            continue
        y -= 0.075
        fig.patches.append(plt.Rectangle((0.02, y - 0.005), 0.012, 0.028, transform=fig.transFigure, color=COLOR[m]))
        fig.text(0.04, y, m, fontsize=10.5, fontweight="bold", color=TEXT)
        for x, key in [(0.17, "zero-shot"), (0.58, "fine-tuned")]:
            fig.text(x, y, textwrap.fill(f"“{results[m][key]}”", 55), fontsize=10, color=TEXT, va="bottom")
    fig.text(0.02, 0.015, "CLIP4Clip picks the closest caption from the training set (retrieval); "
             "GIT and BLIP-2 write a new caption word by word.", fontsize=8.5, color=TEXT2)
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="test video id; random if omitted")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip", nargs="*", default=[], help="e.g. --skip blip2")
    args = ap.parse_args()
    root = cfg("git_base")["data_root"]
    ann = json.load(open(f"{root}/msvd_annotations.json"))
    vid = args.video or random.Random(args.seed).choice(sorted(ann["test"]))
    refs = ann["test"][vid]
    print(f"video: {vid}  ({len(refs)} human captions)")

    results = {}
    steps = [("CLIP4Clip", "clip4clip", lambda: run_clip4clip(vid, root, ann["train"])),
             ("GIT", "git", lambda: run_git(vid, root)),
             ("BLIP-2", "blip2", lambda: run_blip2(vid, root))]
    for name, key, fn in steps:
        if key in args.skip:
            continue
        t = time.time(); results[name] = fn()
        print(f"\n[{name}]  ({time.time() - t:.0f}s)")
        print(f"  zero-shot : {results[name]['zero-shot']}")
        print(f"  fine-tuned: {results[name]['fine-tuned']}")
        if name == "CLIP4Clip":
            print("  fine-tuned top-3 retrieved (caption, similarity):", results[name]["fine-tuned top-3"])

    os.makedirs("results/demo", exist_ok=True)
    make_figure(root, vid, refs, results, f"results/demo/{vid}.png")
    make_gif(root, vid, f"results/demo/{vid}.gif")
    json.dump({"video": vid, "references": refs, "results": results}, open(f"results/demo/{vid}.json", "w"), indent=2)
    print(f"\nsaved results/demo/{vid}.png, .gif, .json")


if __name__ == "__main__":
    main()
