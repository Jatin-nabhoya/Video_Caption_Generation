"""Download raw MSVD + standard splits/captions, extract frames, write annotations.
Run ONCE on Kaggle (Internet ON). Output folder becomes your Kaggle Dataset.
"""
import argparse, glob, json, os, pickle, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("--work", default="/kaggle/working/_build")
p.add_argument("--out", default="/kaggle/working/msvd")
p.add_argument("--num_frames", type=int, default=12)
args = p.parse_args()
os.makedirs(args.work, exist_ok=True)
os.makedirs(f"{args.out}/frames", exist_ok=True)

def sh(cmd):
    print(">>", cmd); subprocess.run(cmd, shell=True, check=True)

# 1) Download: raw videos and standard splits/captions.
sh(f"wget -q -nc -P {args.work} https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar")
sh(f"wget -q -nc -P {args.work} https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip")
sh(f"tar -xf {args.work}/YouTubeClips.tar -C {args.work}")
sh(f"unzip -o -q {args.work}/msvd_data.zip -d {args.work}")

# 2) Captions + splits.
caps = pickle.load(open(f"{args.work}/msvd_data/raw-captions.pkl", "rb"))
splits = {s: open(f"{args.work}/msvd_data/{s}_list.txt").read().split() for s in ["train", "val", "test"]}
print({s: len(v) for s, v in splits.items()})

# 3) Extract evenly spaced JPEG frames.
videos = {os.path.splitext(os.path.basename(v))[0]: v for v in glob.glob(f"{args.work}/**/*.avi", recursive=True)}
print("videos found:", len(videos))

def extract(vid):
    d = f"{args.out}/frames/{vid}"
    if len(glob.glob(f"{d}/*.jpg")) == args.num_frames:
        return vid, True
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", videos[vid]], capture_output=True, text=True)
    dur = float(r.stdout.strip() or 0)
    for i in range(args.num_frames):
        t = (i + 0.5) * dur / args.num_frames
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", videos[vid], "-frames:v", "1", "-vf", "scale=-2:256", "-q:v", "3", f"{d}/{i:02d}.jpg"])
    return vid, len(glob.glob(f"{d}/*.jpg")) == args.num_frames

all_ids = [v for s in splits.values() for v in s]
missing = [v for v in all_ids if v not in videos]
print("ids without a video file:", len(missing))
with ThreadPoolExecutor(8) as ex:
    results = list(ex.map(extract, [v for v in all_ids if v in videos]))
failed = [v for v, ok in results if not ok]
print("frame extraction failed for:", len(failed), failed[:10])

ann = {s: {v: [" ".join(c) if isinstance(c, list) else c for c in caps[v]] for v in ids if v in videos and v not in failed} for s, ids in splits.items()}
json.dump(ann, open(f"{args.out}/msvd_annotations.json", "w"))
print({s: (len(a), sum(len(c) for c in a.values())) for s, a in ann.items()})
shutil.rmtree(args.work)
print("DONE ->", args.out)
