"""Are the differences between our systems real, or noise? Paired tests on the 670 MSVD test videos.

For every comparison (A vs B, same videos):
  * per-video CIDEr and ROUGE-L for both systems (CIDEr IDF computed once on the full test set, so the
    mean of the per-video scores IS the corpus score - the script checks this against the saved results)
  * mean difference B - A
  * 95% confidence interval from a paired bootstrap (resample videos with replacement, 10,000 times)
  * p-value from a paired sign-flip permutation test (10,000 random sign flips of the per-video differences)
  * Holm correction across all comparisons (we run several tests at once)
  * wins / ties / losses per video

CPU only, about 5-10 minutes (Java is needed for the COCO tokenizer, which Kaggle has).
  python scripts/significance.py --data_root /kaggle/input/datasets/jatinnabhoya/video-caption-generation/msvd
"""
import argparse, glob, json, os, sys

import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

R = "results/"
# (idea, label A, preds A, label B, preds B, saved result json for B or None)
COMPARISONS = [
    ("Idea 1 RAG", "Qwen2.5-VL zero-shot", R + "qwen25vl_zeroshot_test_preds.json",
                   "Qwen2.5-VL zero-shot + RAG", R + "qwen25vl_zeroshot_rag_test_preds.json"),
    ("Idea 1 RAG", "Qwen2.5-VL fine-tuned", R + "qwen25vl_3b_lora_v1_best_test_preds.json",
                   "Qwen2.5-VL fine-tuned + RAG", R + "qwen25vl_rag_v1_best_test_preds.json"),
    ("Idea 2 adapter", "GIT baseline (v2)", R + "git_base_ft_v2_best_test_preds.json",
                       "GIT + adapter v1", R + "git_temporal_v1_best_test_preds.json"),
    ("Idea 2 adapter", "GIT + adapter v1, adapter OFF", R + "git_temporal_v1_adapter_off_test_preds.json",
                       "GIT + adapter v1, adapter ON", R + "git_temporal_v1_best_test_preds.json"),
    ("Idea 2 adapter", "GIT baseline (v2)", R + "git_base_ft_v2_best_test_preds.json",
                       "GIT + adapter v2", R + "git_temporal_v2_best_test_preds.json"),
    ("Idea 3 rerank", "BLIP-2 alone (top beam)", R + "blip2_v1_best_test_preds.json",
                      "Rerank (4 models, tuned on val)", R + "rerank/rerank_test_preds.json"),
]


def find_refs(data_root):
    for c in [data_root, "/kaggle/input/datasets/jatinnabhoya/video-caption-generation/msvd",
              "/kaggle/input/video-caption-generation/msvd", "data/msvd"]:
        if c and os.path.exists(os.path.join(c, "msvd_annotations.json")):
            return json.load(open(os.path.join(c, "msvd_annotations.json")))["test"]
    hits = glob.glob("/kaggle/input/**/msvd_annotations.json", recursive=True)
    if hits:
        return json.load(open(hits[0]))["test"]
    sys.exit("msvd_annotations.json not found - pass --data_root")


def per_video_scores(preds, refs_tok, tok, vids):
    """Per-video CIDEr and ROUGE-L x100, in the order of vids."""
    res = tok.tokenize({v: [{"caption": preds[v] if preds[v].strip() else "empty"}] for v in vids})
    gts = {v: refs_tok[v] for v in vids}
    res = {v: res[v] for v in vids}
    _, cider = Cider().compute_score(gts, res)
    _, rouge = Rouge().compute_score(gts, res)
    return np.asarray(cider, float) * 100, np.asarray(rouge, float) * 100


def paired_test(a, b, n_boot, rng):
    d = b - a
    n = len(d)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm = (signs * d).mean(1)
    p = (np.sum(np.abs(perm) >= abs(d.mean()) - 1e-12) + 1) / (n_boot + 1)
    return {"mean_A": float(a.mean()), "mean_B": float(b.mean()), "diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum())}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/significance")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    refs = find_refs(a.data_root)

    comps = []
    for idea, la, fa, lb, fb in COMPARISONS:
        missing = [f for f in (fa, fb) if not os.path.exists(f)]
        if missing:
            print(f"SKIP {la} vs {lb}: missing {missing}")
            continue
        comps.append((idea, la, json.load(open(fa)), fa, lb, json.load(open(fb)), fb))
    if not comps:
        sys.exit("no comparison has both prediction files")

    vids = sorted(set(refs).intersection(*[set(c[2]) & set(c[5]) for c in comps]))
    print(f"{len(vids)} test videos used in every comparison (test set has {len(refs)})")
    tok = PTBTokenizer()
    refs_tok = tok.tokenize({v: [{"caption": c} for c in refs[v]] for v in vids})

    cache, rows = {}, []
    for idea, la, pa, fa, lb, pb, fb in comps:
        for f, p in ((fa, pa), (fb, pb)):
            if f not in cache:
                cache[f] = per_video_scores(p, refs_tok, tok, vids)
                saved = f.replace("_preds.json", ".json")
                if os.path.exists(saved) and saved != f:
                    s = json.load(open(saved)).get("scores", {}).get("CIDEr")
                    if s is not None:
                        flag = "OK" if abs(s - cache[f][0].mean()) < 0.5 else "MISMATCH - check this file"
                        print(f"  check {os.path.basename(f)}: recomputed CIDEr {cache[f][0].mean():.2f} vs saved {s:.2f} -> {flag}")
        for metric, k in (("CIDEr", 0), ("ROUGE-L", 1)):
            r = paired_test(cache[fa][k], cache[fb][k], a.n_boot, rng)
            rows.append({"idea": idea, "A": la, "B": lb, "metric": metric, **r})

    for metric in ("CIDEr", "ROUGE-L"):
        idx = [i for i, r in enumerate(rows) if r["metric"] == metric]
        adj = holm(np.array([rows[i]["p_value"] for i in idx]))
        for i, p in zip(idx, adj):
            rows[i]["p_holm"] = float(p)
            rows[i]["significant"] = bool(p < 0.05)

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"n_videos": len(vids), "n_boot": a.n_boot, "seed": a.seed, "results": rows},
              open(os.path.join(a.out_dir, "significance_test.json"), "w"), indent=2)
    lines = ["| Idea | A | B | Metric | A | B | B - A | 95% CI | p (Holm) | Real? | W / T / L |",
             "|---|---|---|---|---:|---:|---:|---|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['idea']} | {r['A']} | {r['B']} | {r['metric']} | {r['mean_A']:.2f} | {r['mean_B']:.2f} | "
                     f"{r['diff']:+.2f} | [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['p_holm']:.4f} | "
                     f"{'YES' if r['significant'] else 'no'} | {r['wins']} / {r['ties']} / {r['losses']} |")
    open(os.path.join(a.out_dir, "significance_test.md"), "w").write("\n".join(lines) + "\n")

    print(f"\n{'comparison':66s} {'metric':8s} {'B-A':>7s}  {'95% CI':>17s}  {'p(Holm)':>8s}  real?")
    for r in rows:
        name = f"{r['A']} -> {r['B']}"
        print(f"{name[:66]:66s} {r['metric']:8s} {r['diff']:+7.2f}  [{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}]  "
              f"{r['p_holm']:8.4f}  {'YES' if r['significant'] else 'no'}")
    print(f"\nsaved {a.out_dir}/significance_test.json and .md")


if __name__ == "__main__":
    main()
