"""Idea 1 (step 1): retrieve hint captions for every video with the fine-tuned CLIP4Clip.

1. Encode every TRAINING video (12 frames, mean-pooled) and put the vectors in a FAISS index.
2. For each video of train / val / test, find its k nearest training videos.
   Training videos never retrieve themselves or other clips cut from the same YouTube video
   (otherwise the model learns to copy the answer).
3. From each neighbour, keep the one caption that CLIP4Clip finds closest to the QUERY video.

Output: results/rag/hints_<split>.json = {video_id: [caption, ...]}  (+ *_meta.json with scores)
  python scripts/build_rag_index.py --config configs/clip4clip.yaml --k 5
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.msvd import MSVDFrames, make_collate
from src.models.clip4clip import encode_text_list, encode_video_set, load_clip4clip
from src.utils.hub import try_resume


def source_id(vid):
    """'eyhzdC936uk_15_27' -> 'eyhzdC936uk' (the YouTube video the clip was cut from)."""
    return vid.rsplit("_", 2)[0]


class Index:
    """FAISS inner-product index (cosine, vectors are normalized); falls back to plain numpy."""
    def __init__(self, X):
        self.X = X.astype("float32")
        try:
            import faiss
            self.faiss = faiss.IndexFlatIP(X.shape[1]); self.faiss.add(self.X)
        except ImportError:
            print("faiss not installed -> numpy search (same result, fine for 1,200 videos)")
            self.faiss = None

    def search(self, Q, n):
        Q = Q.astype("float32")
        if self.faiss is not None:
            return self.faiss.search(Q, n)
        S = Q @ self.X.T
        I = np.argsort(-S, 1)[:, :n]
        return np.take_along_axis(S, I, 1), I


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/clip4clip.yaml")
    p.add_argument("--k", type=int, default=5, help="neighbours stored per video (training uses the first 3)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out_dir", default="results/rag")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    path, _ = try_resume(cfg["hf_repo"], f"{cfg['run_name']}/best")
    assert path, "fine-tuned CLIP4Clip checkpoint not found on HF Hub"
    processor, model = load_clip4clip(path)
    model.to("cuda").eval()

    def encode(split):
        ds = MSVDFrames(cfg["data_root"], split, processor, cfg["num_frames"], train=False)
        dl = DataLoader(ds, cfg.get("eval_batch_size", 16), num_workers=4, collate_fn=make_collate(processor.tokenizer))
        vids, V = encode_video_set(model, dl, "cuda")
        return ds, vids, V

    train_ds, train_vids, train_V = encode("train")
    index = Index(train_V.cpu().numpy())
    # every training caption, embedded once; grouped by its video
    cap_text, cap_vid = [], []
    for v in train_vids:
        for c in train_ds.caps[v]:
            cap_text.append(c); cap_vid.append(v)
    C = encode_text_list(model, processor.tokenizer, cap_text, "cuda")
    by_vid = {}
    for i, v in enumerate(cap_vid):
        by_vid.setdefault(v, []).append(i)

    os.makedirs(a.out_dir, exist_ok=True)
    for split in a.splits:
        if split == "train":
            vids, V = train_vids, train_V
        else:
            _, vids, V = encode(split)
        sims, idx = index.search(V.cpu().numpy(), a.k + 20)   # extra: some neighbours get filtered out
        hints, meta = {}, {}
        for qi, q in enumerate(vids):
            chosen = []
            for s, ni in zip(sims[qi], idx[qi]):
                nv = train_vids[ni]
                if nv == q or source_id(nv) == source_id(q):     # no self / same-source leakage
                    continue
                rows = by_vid[nv]
                best = rows[int((C[rows] @ V[qi]).argmax())]     # neighbour caption closest to the query video
                chosen.append({"neighbour": nv, "video_sim": round(float(s), 4), "caption": cap_text[best]})
                if len(chosen) == a.k:
                    break
            hints[q] = [c["caption"] for c in chosen]
            meta[q] = chosen
        json.dump(hints, open(os.path.join(a.out_dir, f"hints_{split}.json"), "w"), indent=1)
        json.dump(meta, open(os.path.join(a.out_dir, f"hints_{split}_meta.json"), "w"), indent=1)
        print(f"[{split}] {len(hints)} videos -> {a.out_dir}/hints_{split}.json | example:",
              vids[0], hints[vids[0]][:3])


if __name__ == "__main__":
    main()
