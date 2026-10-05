| Model                                        | Setting           |   BLEU-4 |   METEOR |   ROUGE-L |   CIDEr |    R@1 |    R@5 |   R@10 |    MdR |
|:---------------------------------------------|:------------------|---------:|---------:|----------:|--------:|-------:|-------:|-------:|-------:|
| CLIP4Clip                                    | Zero-shot (ours)  |    17.48 |    25.32 |     50.79 |   39.36 |  37.05 |  64.78 |  74.22 |   3.00 |
| CLIP4Clip                                    | Fine-tuned (ours) |    22.85 |    26.96 |     54.15 |   45.89 |  45.79 |  75.04 |  84.03 |   2.00 |
| GIT                                          | Zero-shot (ours)  |    18.71 |    23.20 |     47.73 |   39.85 | nan    | nan    | nan    | nan    |
| GIT                                          | Fine-tuned (ours) |    54.25 |    38.00 |     75.10 |  101.27 | nan    | nan    | nan    | nan    |
| BLIP-2                                       | Zero-shot (ours)  |    29.12 |    32.96 |     58.22 |   49.07 | nan    | nan    | nan    | nan    |
| BLIP-2                                       | Fine-tuned (ours) |    75.47 |    49.88 |     86.99 |  178.34 | nan    | nan    | nan    | nan    |
| Qwen2.5-VL                                   | Zero-shot (ours)  |    42.54 |    36.89 |     68.94 |   81.88 | nan    | nan    | nan    | nan    |
| Qwen2.5-VL                                   | Fine-tuned (ours) |    67.98 |    44.65 |     82.13 |  146.95 | nan    | nan    | nan    | nan    |
| GIT paper (full-size GIT)                    | published         |    79.50 |   nan    |    nan    |  180.20 | nan    | nan    | nan    | nan    |
| SwinBERT (prior SOTA, as cited in GIT paper) | published         |    58.20 |   nan    |    nan    |  120.60 | nan    | nan    | nan    | nan    |
| CLIP4Clip paper (meanP)                      | published         |   nan    |   nan    |    nan    |  nan    |  46.20 |  76.10 |  84.60 |   2.00 |