"""Extra figures: BLIP-2 vs published, and ALL fine-tuned models vs published MSVD results.
Reuses data + style from compare_results.py. Run after compare_results.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_results import MODEL_COLOR, PAPER_CAPTION, REF_COLORS, bar, load_results

OUT = "results/comparison"
METS = ["BLEU-4", "CIDEr"]          # the two metrics the published MSVD numbers report
os.makedirs(OUT, exist_ok=True)
res = load_results()


def vs_published(series, title, fname, figsize=(8, 4.2)):
    series = [s for s in series if s[1]]
    fig, ax = plt.subplots(figsize=figsize)
    w = 0.8 / len(series); x = np.arange(len(METS))
    for i, (n, r, c, h) in enumerate(series):
        bar(ax, x + (i - (len(series) - 1) / 2) * w, [r.get(m, np.nan) for m in METS], w, c, h, n)
    ax.set_xticks(x, METS); ax.set_ylabel("score (x100)"); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=160); plt.close(fig)
    print("saved", fname)


refs = [(n, r, c, None) for (n, r), c in zip(PAPER_CAPTION.items(), REF_COLORS)]

# fig6: BLIP-2 zero-shot + fine-tuned vs published MSVD results
if ("BLIP-2", "Fine-tuned") in res or ("BLIP-2", "Zero-shot") in res:
    vs_published([("BLIP-2 zero-shot (ours)", res.get(("BLIP-2", "Zero-shot")), MODEL_COLOR["BLIP-2"], "//"),
                  ("BLIP-2 fine-tuned (ours)", res.get(("BLIP-2", "Fine-tuned")), MODEL_COLOR["BLIP-2"], None)] + refs,
                 "BLIP-2 on MSVD test: ours vs. published", "fig6_blip2_vs_published.png")
    ft = res.get(("BLIP-2", "Fine-tuned"))
    if ft:
        rel = [{"Reference": n, **{f"ours / ref ({k})": f"{100 * ft[k] / v:.0f}%" for k, v in r.items()}}
               for n, r in PAPER_CAPTION.items()]
        pd.DataFrame(rel).to_markdown(f"{OUT}/blip2_vs_published.md", index=False)

# fig7: every fine-tuned model of ours vs published MSVD results
ours = [(f"{m} fine-tuned (ours)", res.get((m, "Fine-tuned")), MODEL_COLOR[m], None) for m in MODEL_COLOR]
vs_published(ours + refs, "All fine-tuned models vs. published results (MSVD test)",
             "fig7_all_models_vs_published.png", figsize=(9.5, 4.4))
