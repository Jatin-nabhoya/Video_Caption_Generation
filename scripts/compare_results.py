"""Build all comparison tables + figures from results/*.json and the training histories on HF Hub.
Output: results/comparison/ (PNG figures, CSV/Markdown tables, qualitative examples).
Add BLIP-2 later by filling its entries in FILES / RUNS - everything else updates automatically."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- inputs
FILES = {  # (model, setting) -> test-set result file. Missing files are skipped.
    ("CLIP4Clip", "Zero-shot"):  "results/clip4clip_zeroshot_test.json",
    ("CLIP4Clip", "Fine-tuned"): "results/clip4clip_meanP_v1_best_test.json",
    ("GIT", "Zero-shot"):        "results/git_base_zeroshot_test.json",
    ("GIT", "Fine-tuned"):       "results/git_base_ft_v2_best_test.json",
    ("BLIP-2", "Zero-shot"):     "results/blip2_zeroshot_test.json",
    ("BLIP-2", "Fine-tuned"):    "results/blip2_v1_best_test.json",
    ("Qwen2.5-VL", "Zero-shot"):  "results/qwen25vl_zeroshot_test.json",
    ("Qwen2.5-VL", "Fine-tuned"): "results/qwen25vl_3b_lora_v1_best_test.json",
}
RUNS = {"CLIP4Clip": "clip4clip_meanP_v1", "GIT": "git_base_ft_v2", "BLIP-2": "blip2_v1", "Qwen2.5-VL": "qwen25vl_3b_lora_v1"}  # training histories on HF
HF_REPO = "jatinnabhoya/Video_Caption_Generation"

# Published MSVD test numbers (scores x100)
PAPER_CAPTION = {  # source: GIT paper (Wang et al. 2022), Table 5 - MSVD
    "GIT paper (full-size GIT)": {"BLEU-4": 79.5, "CIDEr": 180.2},
    "SwinBERT (prior SOTA, as cited in GIT paper)": {"BLEU-4": 58.2, "CIDEr": 120.6},
}
PAPER_RETRIEVAL = {  # source: CLIP4Clip (Luo et al. 2021), MSVD text-to-video, meanP ViT-B/32
    "CLIP4Clip paper (meanP)": {"R@1": 46.2, "R@5": 76.1, "R@10": 84.6, "MdR": 2.0},
}
CAP_METRICS = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
RET_METRICS = ["R@1", "R@5", "R@10"]

# ---------------------------------------------------------------- style (validated categorical slots 1-3)
MODEL_COLOR = {"CLIP4Clip": "#2a78d6", "GIT": "#eb6834", "BLIP-2": "#1baf7a", "Qwen2.5-VL": "#7c5cd6"}
REF_COLORS = ["#8a8984", "#52514e"]           # neutral grays for published reference numbers
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})
plt.rcParams["axes.grid.axis"] = "y"


def bar(ax, x, h, w, color, hatch=None, label=None):
    """Bar with a 2px surface gap and a value label on top (direct labels = no color-only reading)."""
    b = ax.bar(x, h, w, color=SURFACE if hatch else color, edgecolor=color if hatch else SURFACE,
               linewidth=2 if not hatch else 1.5, hatch=hatch, label=label)
    for r, v in zip(b, np.atleast_1d(h)):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TEXT2)
    return b


def load_results():
    rows = {}
    for (model, setting), f in FILES.items():
        if os.path.exists(f):
            r = json.load(open(f))
            rows[(model, setting)] = {**r["scores"], **r.get("retrieval", {}), "n_videos": r.get("n_videos")}
        else:
            print(f"(skip - not found yet) {f}")
    return rows


def load_histories(use_hf):
    hist = {}
    if not use_hf:
        return hist
    from huggingface_hub import hf_hub_download
    for model, run in RUNS.items():
        try:
            p = hf_hub_download(HF_REPO, f"{run}/last/state.json", force_download=True)
            hist[model] = pd.DataFrame(json.load(open(p))["history"])
        except Exception as e:
            print(f"(skip history) {model}/{run}: {type(e).__name__}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/comparison")
    ap.add_argument("--no_hf", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = load_results()
    hist = load_histories(not args.no_hf)
    models = [m for m in MODEL_COLOR if any(k[0] == m for k in res)]

    # ---------------- table 1: everything in one place
    table = []
    for (model, setting), r in res.items():
        table.append({"Model": model, "Setting": setting + " (ours)", **{k: r.get(k) for k in CAP_METRICS + RET_METRICS + ["MdR"]}})
    for name, r in {**PAPER_CAPTION, **PAPER_RETRIEVAL}.items():
        table.append({"Model": name, "Setting": "published", **r})
    df = pd.DataFrame(table)
    df.to_csv(f"{args.out}/comparison_table.csv", index=False)
    open(f"{args.out}/comparison_table.md", "w").write(df.to_markdown(index=False, floatfmt=".2f"))

    # ---------------- table 2: gain from fine-tuning + share of the published result
    gain = []
    for m in models:
        zs, ft = res.get((m, "Zero-shot")), res.get((m, "Fine-tuned"))
        if zs and ft:
            gain.append({"Model": m, **{f"Δ {k}": round(ft[k] - zs[k], 2) for k in CAP_METRICS}})
    if gain:
        pd.DataFrame(gain).to_markdown(f"{args.out}/finetuning_gain.md", index=False)
    if ("GIT", "Fine-tuned") in res:
        ft = res[("GIT", "Fine-tuned")]
        rel = [{"Reference": n, **{f"ours / ref ({k})": f"{100 * ft[k] / v:.0f}%" for k, v in r.items()}}
               for n, r in PAPER_CAPTION.items()]
        pd.DataFrame(rel).to_markdown(f"{args.out}/git_vs_paper.md", index=False)

    # ---------------- figure 1: GIT vs published results (BLEU-4, CIDEr)
    if ("GIT", "Fine-tuned") in res or ("GIT", "Zero-shot") in res:
        mets = ["BLEU-4", "CIDEr"]
        series = [("GIT-base zero-shot (ours)", res.get(("GIT", "Zero-shot")), MODEL_COLOR["GIT"], "//"),
                  ("GIT-base fine-tuned (ours)", res.get(("GIT", "Fine-tuned")), MODEL_COLOR["GIT"], None)]
        series += [(n, r, c, None) for (n, r), c in zip(PAPER_CAPTION.items(), REF_COLORS)]
        series = [s for s in series if s[1]]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        w = 0.8 / len(series); x = np.arange(len(mets))
        for i, (n, r, c, h) in enumerate(series):
            bar(ax, x + (i - (len(series) - 1) / 2) * w, [r.get(m, np.nan) for m in mets], w, c, h, n)
        ax.set_xticks(x, mets); ax.set_ylabel("score (x100)")
        ax.set_title("GIT on MSVD test: ours vs. published")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig1_git_vs_paper.png", dpi=160); plt.close(fig)

    # ---------------- figure 2: all models, zero-shot vs fine-tuned, one panel per metric
    if models:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
        x = np.arange(len(models)); w = 0.38
        for ax, met in zip(axes, CAP_METRICS):
            for j, m in enumerate(models):
                zs, ft = res.get((m, "Zero-shot")), res.get((m, "Fine-tuned"))
                bar(ax, [x[j] - w / 2], [zs[met] if zs else np.nan], w, MODEL_COLOR[m], "//")
                bar(ax, [x[j] + w / 2], [ft[met] if ft else np.nan], w, MODEL_COLOR[m])
            ax.set_xticks(x, models); ax.set_title(met)
        axes[0].set_ylabel("score (x100)")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=SURFACE, edgecolor=TEXT2, hatch="//", label="Zero-shot"),
                            Patch(facecolor=TEXT2, label="Fine-tuned")], loc="upper right", ncol=2)
        fig.suptitle("Captioning on MSVD test: zero-shot vs. fine-tuned", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{args.out}/fig2_models_captioning.png", dpi=160); plt.close(fig)

    # ---------------- figure 3: CLIP4Clip retrieval vs paper
    c_zs, c_ft = res.get(("CLIP4Clip", "Zero-shot")), res.get(("CLIP4Clip", "Fine-tuned"))
    if (c_zs or c_ft) and "R@1" in (c_ft or c_zs):
        series = [("CLIP4Clip zero-shot (ours)", c_zs, MODEL_COLOR["CLIP4Clip"], "//"),
                  ("CLIP4Clip fine-tuned (ours)", c_ft, MODEL_COLOR["CLIP4Clip"], None)]
        series += [(n, r, REF_COLORS[1], None) for n, r in PAPER_RETRIEVAL.items()]
        series = [s for s in series if s[1]]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        w = 0.8 / len(series); x = np.arange(len(RET_METRICS))
        for i, (n, r, c, h) in enumerate(series):
            bar(ax, x + (i - (len(series) - 1) / 2) * w, [r[m] for m in RET_METRICS], w, c, h, n)
        ax.set_xticks(x, RET_METRICS); ax.set_ylabel("recall (%)"); ax.set_ylim(0, 100)
        ax.set_title("CLIP4Clip text-to-video retrieval on MSVD test: ours vs. paper")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig3_clip4clip_retrieval_vs_paper.png", dpi=160); plt.close(fig)

    # ---------------- figure 4: validation CIDEr per epoch (same metric -> one axis)
    if hist:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for m, h in hist.items():
            ax.plot(h["epoch"], h["CIDEr"], color=MODEL_COLOR[m], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=m)
            ax.annotate(f"{m} {h['CIDEr'].iloc[-1]:.1f}", (h["epoch"].iloc[-1], h["CIDEr"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT2)
        ax.set_xlabel("epoch"); ax.set_ylabel("validation CIDEr (x100)")
        ax.set_title("Validation CIDEr during fine-tuning"); ax.legend(loc="lower right")
        ax.xaxis.get_major_locator().set_params(integer=True)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig4_val_cider_curves.png", dpi=160); plt.close(fig)

        # ---------------- figure 5: training loss, one panel per model (different losses -> never one axis)
        fig, axes = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 3.4), squeeze=False)
        for ax, (m, h) in zip(axes[0], hist.items()):
            ax.plot(h["epoch"], h["train_loss"], color=MODEL_COLOR[m], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.set_title(f"{m} training loss"); ax.set_xlabel("epoch")
            ax.xaxis.get_major_locator().set_params(integer=True)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig5_train_loss.png", dpi=160); plt.close(fig)
        for m, h in hist.items():
            h.to_csv(f"{args.out}/history_{m}.csv", index=False)

    # ---------------- qualitative examples: same test videos, every model's caption
    preds = {k: json.load(open(f.replace(".json", "_preds.json")))
             for k, f in FILES.items() if k[1] == "Fine-tuned" and os.path.exists(f.replace(".json", "_preds.json"))}
    if preds:
        ann = None
        for cand in ["/kaggle/input", "."]:
            for r, _, fs in os.walk(cand, followlinks=True):
                if "msvd_annotations.json" in fs:
                    ann = json.load(open(os.path.join(r, "msvd_annotations.json")))["test"]; break
            if ann: break
        common = sorted(set.intersection(*[set(p) for p in preds.values()]))[:12]
        ex = [{"video": v, "reference": (ann[v][0] if ann else ""), **{f"{m} (fine-tuned)": p[v] for (m, _), p in preds.items()}}
              for v in common]
        pd.DataFrame(ex).to_markdown(f"{args.out}/qualitative_examples.md", index=False)

    print(df.to_string(index=False))
    print("\nsaved to", args.out, "->", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
