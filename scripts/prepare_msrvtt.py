"""Build MSR-VTT from its ORIGINAL release files (Xu et al., CVPR 2016).
Run ONCE on Kaggle (Settings -> Internet ON). The output folder becomes a Kaggle Dataset.

Original files (same names as the 2016 release):
  train_val_videos.zip, test_videos.zip, train_val_annotation.zip, test_videodatainfo.json.zip
They are fetched from the MediaFire folder most papers link to, or from --zips_dir if you
uploaded them yourself. The official split (6,513 / 497 / 2,990) comes from the JSON "split" field.

Output (same layout as our MSVD data):
  <out>/frames/<video_id>/00.jpg ... 11.jpg
  <out>/annotations.json        {"train"|"val"|"test": {video_id: [captions]}}
  <out>/original_annotations/   the untouched original JSON files
  <out>/data_card.json          sources, sha256, counts, anything missing
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

FOLDER_KEY = "h14iarbs62e7p"   # https://www.mediafire.com/folder/h14iarbs62e7p/shared
REQUIRED = ["train_val_videos.zip", "test_videos.zip", "train_val_annotation.zip", "test_videodatainfo.json.zip"]
EXPECTED = {"train": 6513, "val": 497, "test": 2990}      # Xu et al. 2016, Table/Sec. 3
SPLIT_NAME = {"train": "train", "validate": "val", "test": "test"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def mediafire_list(folder_key):
    """Return {filename: (quickkey, size)} for every file in the folder and its sub-folders."""
    api = "https://www.mediafire.com/api/1.5/folder/get_content.php"
    out, stack = {}, [folder_key]
    while stack:
        key = stack.pop()
        for ctype in ("files", "folders"):
            chunk = 1
            while True:
                r = requests.get(api, params={"folder_key": key, "content_type": ctype, "chunk": chunk,
                                              "response_format": "json"}, headers=UA, timeout=60)
                r.raise_for_status()
                fc = r.json()["response"]["folder_content"]
                for f in fc.get("files", []):
                    out[f["filename"]] = (f["quickkey"], int(f.get("size", 0)))
                for d in fc.get("folders", []):
                    stack.append(d["folderkey"])
                if fc.get("more_chunks") != "yes":
                    break
                chunk += 1
    return out


def mediafire_direct_link(html):
    """Pull the real download URL out of a MediaFire file page."""
    m = re.search(r'id="downloadButton"[^>]*href="(https?://[^"]+)"', html) or \
        re.search(r'href="(https?://download\d*\.mediafire\.com/[^"]+)"', html)
    if not m:
        raise RuntimeError("could not find the download link on the MediaFire page")
    return m.group(1)


def download(url, path, size=0):
    if os.path.exists(path) and (size == 0 or os.path.getsize(path) == size):
        print(f"already have {os.path.basename(path)}"); return
    tmp = path + ".part"
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 22):
                f.write(block)
    if size and os.path.getsize(tmp) != size:
        raise RuntimeError(f"{os.path.basename(path)}: got {os.path.getsize(tmp)} bytes, expected {size}")
    os.replace(tmp, path)
    print(f"downloaded {os.path.basename(path)} ({os.path.getsize(path) / 1e9:.2f} GB)")


def fetch_from_mediafire(work):
    files = mediafire_list(FOLDER_KEY)
    print("files in the MediaFire folder:", sorted(files))
    missing = [n for n in REQUIRED if n not in files]
    if missing:
        raise RuntimeError(f"not in the MediaFire folder: {missing}. Upload them yourself and use --zips_dir.")
    for name in REQUIRED:
        qk, size = files[name]
        page = requests.get(f"https://www.mediafire.com/file/{qk}/{name}/file", headers=UA, timeout=60)
        page.raise_for_status()
        download(mediafire_direct_link(page.text), os.path.join(work, name), size)
    return {n: os.path.join(work, n) for n in REQUIRED}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_split_json(paths):
    """Merge train_val_videodatainfo.json + test_videodatainfo.json into {split: {vid: [captions]}}."""
    ann, n_sent = {s: {} for s in EXPECTED}, 0
    for p in paths:
        d = json.load(open(p))
        split_of = {v["video_id"]: SPLIT_NAME[v["split"]] for v in d["videos"]}
        for v, s in split_of.items():
            ann[s].setdefault(v, [])
        for s in d["sentences"]:
            ann[split_of[s["video_id"]]][s["video_id"]].append(s["caption"].strip())
            n_sent += 1
    return ann, n_sent


def extract(job):
    vid, src, out_dir, n = job
    d = os.path.join(out_dir, vid)
    if len(glob.glob(f"{d}/*.jpg")) == n:
        return vid, True
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                       capture_output=True, text=True)
    dur = float(r.stdout.strip() or 0)
    for i in range(n):
        t = (i + 0.5) * dur / n
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                        "-vf", "scale=-2:256", "-q:v", "3", f"{d}/{i:02d}.jpg"])
    return vid, len(glob.glob(f"{d}/*.jpg")) == n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/tmp/msrvtt_build")      # big temporary files, not saved by Kaggle
    p.add_argument("--out", default="/kaggle/working/msrvtt")
    p.add_argument("--zips_dir", default=None, help="folder that already holds the 4 original zips")
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_count_check", action="store_true", help="testing only: skip the official-count checks")
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True); os.makedirs(f"{a.out}/frames", exist_ok=True)

    # 1) original zips (Kaggle unzips uploaded .zip files, so --zips_dir may hold zips OR their contents)
    if a.zips_dir:
        zips = {n: os.path.join(a.zips_dir, n) for n in REQUIRED if os.path.isfile(os.path.join(a.zips_dir, n))}
        print(f"--zips_dir: {len(zips)} zip(s) found, the rest is read as already-unzipped files")
    else:
        zips = fetch_from_mediafire(a.work)
    card = {"dataset": "MSR-VTT (Xu et al., CVPR 2016)", "source": a.zips_dir or f"mediafire folder {FOLDER_KEY}",
            "files": {n: {"bytes": os.path.getsize(z), "sha256": sha256(z)} for n, z in zips.items()}}

    # 2) unzip
    ext = os.path.join(a.work, "extracted")
    os.makedirs(ext, exist_ok=True)
    for n, z in zips.items():
        print("unzipping", n)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ext)
        if not a.zips_dir:
            os.remove(z)          # free disk; sha256 is already in the data card
    roots = [ext] + ([a.zips_dir] if a.zips_dir else [])

    def find(pattern):
        """Every file matching pattern under the search roots, one per file name."""
        hits = {}
        for r in roots:
            for f in glob.glob(f"{r}/**/{pattern}", recursive=True):
                if os.path.isfile(f):
                    hits.setdefault(os.path.basename(f), f)
        return hits

    # 3) original annotations -> splits, checked against the paper
    found = find("*videodatainfo.json")
    need = ["train_val_videodatainfo.json", "test_videodatainfo.json"]
    assert all(n in found for n in need), f"annotation json missing, found only: {sorted(found)}"
    jsons = [found[n] for n in need]
    os.makedirs(f"{a.out}/original_annotations", exist_ok=True)
    for j in jsons:
        shutil.copy(j, f"{a.out}/original_annotations/")
    ann, n_sent = load_split_json(jsons)
    counts = {s: len(v) for s, v in ann.items()}
    print("videos per split:", counts, "| captions:", n_sent)
    if not a.no_count_check:
        assert counts == EXPECTED, f"split sizes {counts} != original {EXPECTED}"
        assert n_sent == 200000, f"{n_sent} captions, the original release has 200,000"

    # 4) videos
    videos = {os.path.splitext(n)[0]: f for n, f in find("*.mp4").items()}
    all_ids = [v for s in ann.values() for v in s]
    no_file = [v for v in all_ids if v not in videos]
    print(f"video files: {len(videos)} | ids without a file: {len(no_file)}")

    # 5) frames
    jobs = [(v, videos[v], f"{a.out}/frames", a.num_frames) for v in all_ids if v in videos]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(extract, jobs))
    failed = [v for v, ok in results if not ok]
    print("frame extraction failed for:", len(failed), failed[:10])

    # 6) annotations + data card
    bad = set(no_file) | set(failed)
    final = {s: {v: c for v, c in vids.items() if v not in bad} for s, vids in ann.items()}
    json.dump(final, open(f"{a.out}/annotations.json", "w"))
    card.update(split_videos_original=counts, captions_original=n_sent,
                split_videos_kept={s: len(v) for s, v in final.items()},
                captions_kept=sum(len(c) for v in final.values() for c in v.values()),
                missing_video_file=no_file, frame_extraction_failed=failed, num_frames=a.num_frames)
    json.dump(card, open(f"{a.out}/data_card.json", "w"), indent=2)
    print(json.dumps({k: card[k] for k in ("split_videos_kept", "captions_kept")}))
    shutil.rmtree(a.work, ignore_errors=True)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
