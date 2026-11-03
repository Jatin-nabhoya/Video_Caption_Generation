"""Build VATEX from its ORIGINAL release (Wang et al., ICCV 2019).
The authors released annotations only; each clip is a 10-second window of a YouTube video,
named <youtube_id>_<start>_<end>. So we read the official JSON files and cut the frames
straight from YouTube at those exact times.

Run this on YOUR LAPTOP (YouTube usually blocks cloud machines such as Kaggle):
    pip install yt-dlp requests        # and install ffmpeg (brew / apt / ffmpeg.org)
    python download_vatex.py --out vatex                       # test + val + all of train
    python download_vatex.py --out vatex --limit_train 10000   # smaller train set
It is safe to stop and restart: finished clips are skipped.
If YouTube asks you to sign in / "confirm you're not a bot": export cookies once, then pass the file
    yt-dlp --cookies-from-browser chrome --cookies cookies.txt --skip-download "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    python download_vatex.py --out vatex --cookies cookies.txt

Output (same layout as our MSVD data):
  <out>/frames/<videoID>/00.jpg ... 11.jpg
  <out>/annotations.json        {"train"|"val"|"test": {videoID: [10 English captions]}}
  <out>/original_annotations/   the untouched official JSON files
  <out>/data_card.json          counts per split + every clip that could not be downloaded, and why
"""
import argparse, glob, json, os, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yt_dlp

BASE = "https://eric-xw.github.io/vatex-website/data/"
FILES = {"train": "vatex_training_v1.0.json",              # 25,991 clips
         "val": "vatex_validation_v1.0.json",              # 3,000 clips
         "test": "vatex_public_test_english_v1.1.json"}    # 6,000 clips, English captions (v1.1)
EXPECTED = {"train": 25991, "val": 3000, "test": 6000}


def parse_id(video_id):
    """'Ptf_2VRj-V0_000122_000132' -> ('Ptf_2VRj-V0', 122, 132). YouTube ids may contain '_'."""
    yt, start, end = video_id.rsplit("_", 2)
    return yt, int(start), int(end)


def get_annotations(out):
    os.makedirs(f"{out}/original_annotations", exist_ok=True)
    ann = {}
    for split, name in FILES.items():
        path = f"{out}/original_annotations/{name}"
        if not os.path.exists(path):
            print("downloading", BASE + name)
            r = requests.get(BASE + name, timeout=120); r.raise_for_status()
            open(path, "wb").write(r.content)
        rows = json.load(open(path))
        assert all("enCap" in r for r in rows), f"{name}: some clips have no English captions ('enCap')"
        ann[split] = {r["videoID"]: [c.strip() for c in r["enCap"]] for r in rows}
        assert len(ann[split]) == EXPECTED[split], f"{split}: {len(ann[split])} clips, official is {EXPECTED[split]}"
        assert all(len(c) == 10 for c in ann[split].values()), f"{split}: not 10 English captions per clip"
    print({s: len(v) for s, v in ann.items()})
    return ann


class _Silent:
    """yt-dlp prints every error in red even with quiet=True; we record reasons in status.json instead."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def stream_url(yt_id, height, cookies):
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "logger": _Silent(),
            "format": f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/b[height<={height}]/b"}
    if cookies and os.path.isfile(cookies):
        opts["cookiefile"] = cookies                 # cookies.txt exported once (fast, no keychain prompts)
    elif cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={yt_id}", download=False)
    fmt = info["requested_formats"][0] if info.get("requested_formats") else info
    return fmt["url"], fmt.get("http_headers") or info.get("http_headers") or {}


def cut_frames(src, headers, start, end, n, d):
    """n frames at the centres of n equal slices of [start, end]; one ffmpeg read of the window."""
    dur = end - start
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",   # fps filter centres each frame in its slot
            "-vf", f"fps={n}/{dur},scale=-2:256", "-frames:v", str(n), "-q:v", "3",
            "-start_number", "0", f"{d}/%02d.jpg"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.stderr.strip()


def process(video_id, frames_dir, n, height, cookies, min_frames):
    d = os.path.join(frames_dir, video_id)
    if len(glob.glob(f"{d}/*.jpg")) >= min_frames:
        return video_id, "ok", ""
    os.makedirs(d, exist_ok=True)
    yt, start, end = parse_id(video_id)
    for attempt in range(3):
        try:
            url, headers = stream_url(yt, height, cookies)
            err = cut_frames(url, headers, start, end, n, d)
            got = len(glob.glob(f"{d}/*.jpg"))
            if got >= min_frames:
                return video_id, "ok", ""
            reason = f"only {got} frames: {err[:200]}"
        except yt_dlp.utils.DownloadError as e:
            msg = str(e).replace("ERROR: ", "")[:200]
            low = msg.lower()
            if "sign in to confirm" in low or "please sign in" in low or "429" in low:
                time.sleep(20)                       # YouTube bot check: slow down, keep for a re-run
                reason = "blocked by YouTube bot check: " + msg
                continue
            if any(k in low for k in ("unavailable", "private", "removed", "terminated", "copyright", "age")):
                if not os.listdir(d):
                    os.rmdir(d)
                return video_id, "missing", msg      # gone for good, no retry
            reason = msg
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:200]}"
        time.sleep(2 * (attempt + 1))
    return video_id, "failed", reason


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="vatex")
    p.add_argument("--splits", nargs="+", default=["test", "val", "train"])   # eval splits first
    p.add_argument("--limit_train", type=int, default=0, help="0 = all 25,991 train clips")
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--min_frames", type=int, default=8, help="keep a clip with at least this many frames")
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cookies_from_browser", default=None, help="chrome | firefox | safari | edge")
    p.add_argument("--cookies", default=None, help="cookies.txt exported once with yt-dlp (recommended)")
    a = p.parse_args()
    ann = get_annotations(a.out)
    frames_dir = f"{a.out}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    status_path = f"{a.out}/status.json"
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    lock = threading.Lock()

    for split in a.splits:
        ids = sorted(ann[split])
        if split == "train" and a.limit_train:
            ids = ids[:a.limit_train]          # sorted -> the same subset every time
        todo = [v for v in ids if status.get(v, {}).get("state") not in ("ok", "missing")]
        print(f"[{split}] {len(ids)} clips, {len(ids) - len(todo)} already done, {len(todo)} to go")
        done = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(process, v, frames_dir, a.num_frames, a.height, a.cookies or a.cookies_from_browser,
                              a.min_frames)
                    for v in todo]
            for f in as_completed(futs):
                vid, state, reason = f.result()
                with lock:
                    status[vid] = {"split": split, "state": state, "reason": reason}
                    done += 1
                    if done % 25 == 0 or done == len(todo):
                        json.dump(status, open(status_path, "w"))
                        n_ok = sum(1 for v in ids if status.get(v, {}).get("state") == "ok")
                        st = [status.get(v, {}).get("state") for v in ids]
                        print(f"[{split}] {done}/{len(todo)} processed | ok {n_ok} | gone {st.count('missing')} "
                              f"| failed {st.count('failed')} (re-run retries these)", flush=True)
        json.dump(status, open(status_path, "w"))

    # annotations only for clips that have frames
    final, card = {}, {"dataset": "VATEX v1.1 (Wang et al., ICCV 2019)", "annotations": BASE,
                       "videos": "cut from YouTube at the official timestamps", "num_frames": a.num_frames,
                       "min_frames": a.min_frames, "splits": {}}
    for split in a.splits:
        ids = sorted(ann[split])
        if split == "train" and a.limit_train:
            ids = ids[:a.limit_train]
        ok = [v for v in ids if status.get(v, {}).get("state") == "ok"]
        final[split] = {v: ann[split][v] for v in ok}
        card["splits"][split] = {"official": EXPECTED[split], "attempted": len(ids), "kept": len(ok),
                                 "missing": sorted(v for v in ids if status.get(v, {}).get("state") == "missing"),
                                 "failed": sorted(v for v in ids if status.get(v, {}).get("state") == "failed")}
    json.dump(final, open(f"{a.out}/annotations.json", "w"))
    json.dump(card, open(f"{a.out}/data_card.json", "w"), indent=2)
    print({s: (c["kept"], c["attempted"]) for s, c in card["splits"].items()})
    print("DONE ->", a.out, "| re-run the same command to retry clips marked 'failed'")


if __name__ == "__main__":
    main()
