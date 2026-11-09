"""Sampled MBR (minimum Bayes risk) decoding for captioning, a.k.a. "crowd sampling"
(Suzgun et al. 2022: +4.3 CIDEr over beam search with BLIP on COCO, 16 samples, temperature 0.7).

For each video: sample N captions, then output the hypothesis with the highest average CIDEr-D
against the sampled captions (the one the "crowd" agrees with most). The beam-search caption is
added as one extra hypothesis, so MBR can always fall back to it.
"""
import re
from collections import defaultdict

import torch
from tqdm import tqdm

from src.models.blip2_video import video_prefix


def norm(c):
    return " ".join(re.findall(r"[a-z0-9']+", c.lower()))


@torch.no_grad()
def sample_candidates(model, tokenizer, loader, device, n=16, temperature=0.7, top_p=1.0, max_new_tokens=20,
                      prefix_fn=None):
    """{video_id: [n sampled captions]} from the BLIP-2 language model."""
    model.eval()
    prefix_fn = prefix_fn or (lambda b: video_prefix(model, b["pixel_values"].to(device)))
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
    bos = tokenizer.bos_token_id
    out = {}
    for b in tqdm(loader, desc=f"sample n={n} T={temperature}", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            prefix = prefix_fn(b)
            lm = model.language_model
            bos_emb = lm.get_input_embeddings()(torch.full((prefix.shape[0], 1), bos, device=device))
            inputs = torch.cat([prefix.to(bos_emb.dtype), bos_emb], 1)
            mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
            ids = lm.generate(inputs_embeds=inputs, attention_mask=mask, do_sample=True, temperature=temperature,
                              top_p=top_p, num_return_sequences=n, max_new_tokens=max_new_tokens,
                              eos_token_id=[nl, tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id)
        texts = [t.strip() for t in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        for j, vid in enumerate(b["video_ids"]):          # generate() returns the n samples of input j together
            out[vid] = texts[j * n:(j + 1) * n]
    return out


def mbr_select(samples, extra=None):
    """samples: {vid: [captions]} (the evidence and hypotheses); extra: {vid: caption} added as a hypothesis.
    Utility = CIDEr-D of a hypothesis against the samples (leave-one-out for the samples themselves).
    Returns ({vid: chosen caption}, {vid: index of chosen hypothesis, -1 = the extra caption})."""
    from pycocoevalcap.cider.cider import Cider
    gts, res, keys = {}, {}, []
    for v, caps in samples.items():
        caps_n = [norm(c) for c in caps]
        hyps = [(i, c) for i, c in enumerate(caps)]
        if extra and v in extra:
            hyps.append((-1, extra[v]))
        for i, h in hyps:
            refs = [caps_n[j] for j in range(len(caps)) if j != i and caps_n[j]]
            if not refs or not norm(h):
                continue
            key = f"{v}#{i}"
            gts[key], res[key] = refs, [norm(h)]
            keys.append((key, v, i, h))
    _, scores = Cider().compute_score(gts, res)
    best = {}
    for (key, v, i, h), s in zip(keys, scores):
        if v not in best or s > best[v][0]:
            best[v] = (float(s), i, h)
    chosen = {v: best[v][2] if v in best else (extra or {}).get(v, samples[v][0]) for v in samples}
    which = {v: best[v][1] if v in best else -1 for v in samples}
    return chosen, which


def pick_rate(which):
    """Share of videos where MBR kept the beam caption vs a sampled one."""
    counts = defaultdict(int)
    for i in which.values():
        counts["beam" if i == -1 else "sample"] += 1
    total = max(1, sum(counts.values()))
    return {k: round(v / total, 3) for k, v in counts.items()}
