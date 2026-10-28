| Idea | A | B | Metric | A | B | B - A | 95% CI | p (Holm) | Real? | W / T / L |
|---|---|---|---|---:|---:|---:|---|---:|---|---|
| Idea 1 RAG | Qwen2.5-VL zero-shot | Qwen2.5-VL zero-shot + RAG | CIDEr | 81.88 | 95.64 | +13.76 | [+9.16, +18.34] | 0.0006 | YES | 328 / 147 / 195 |
| Idea 1 RAG | Qwen2.5-VL zero-shot | Qwen2.5-VL zero-shot + RAG | ROUGE-L | 68.94 | 72.16 | +3.22 | [+2.10, +4.36] | 0.0006 | YES | 302 / 173 / 195 |
| Idea 1 RAG | Qwen2.5-VL fine-tuned | Qwen2.5-VL fine-tuned + RAG | CIDEr | 146.95 | 152.85 | +5.90 | [+1.73, +10.27] | 0.0304 | YES | 171 / 359 / 140 |
| Idea 1 RAG | Qwen2.5-VL fine-tuned | Qwen2.5-VL fine-tuned + RAG | ROUGE-L | 82.13 | 83.50 | +1.38 | [+0.56, +2.23] | 0.0045 | YES | 154 / 408 / 108 |
| Idea 2 adapter | GIT baseline (v2) | GIT + adapter v1 | CIDEr | 101.27 | 104.38 | +3.10 | [-2.29, +8.49] | 0.5336 | no | 251 / 172 / 247 |
| Idea 2 adapter | GIT baseline (v2) | GIT + adapter v1 | ROUGE-L | 75.10 | 75.12 | +0.02 | [-1.06, +1.07] | 1.0000 | no | 198 / 265 / 207 |
| Idea 2 adapter | GIT + adapter v1, adapter OFF | GIT + adapter v1, adapter ON | CIDEr | 104.31 | 104.38 | +0.07 | [+0.00, +0.21] | 1.0000 | no | 1 / 669 / 0 |
| Idea 2 adapter | GIT + adapter v1, adapter OFF | GIT + adapter v1, adapter ON | ROUGE-L | 75.10 | 75.12 | +0.03 | [+0.00, +0.08] | 1.0000 | no | 1 / 669 / 0 |
| Idea 2 adapter | GIT baseline (v2) | GIT + adapter v2 | CIDEr | 101.27 | 105.14 | +3.87 | [-1.76, +9.52] | 0.5336 | no | 255 / 176 / 239 |
| Idea 2 adapter | GIT baseline (v2) | GIT + adapter v2 | ROUGE-L | 75.10 | 74.32 | -0.79 | [-1.99, +0.41] | 0.6170 | no | 204 / 240 / 226 |
| Idea 3 rerank | BLIP-2 alone (top beam) | Rerank (4 models, tuned on val) | CIDEr | 178.34 | 168.82 | -9.52 | [-14.90, -4.41] | 0.0025 | YES | 174 / 315 / 181 |
| Idea 3 rerank | BLIP-2 alone (top beam) | Rerank (4 models, tuned on val) | ROUGE-L | 86.99 | 85.72 | -1.28 | [-2.27, -0.30] | 0.0360 | YES | 139 / 391 / 140 |
