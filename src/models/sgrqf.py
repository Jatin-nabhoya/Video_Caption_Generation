"""SG-RQF: Similarity-Gated Retrieval Q-Former (our novel method, built on BLIP-2 OPT-2.7B).

Plain BLIP-2 (our baseline, src/models/blip2_video.py): per frame, 32 learned queries cross-attend
to the frozen ViT features; the 6 x 32 outputs are the soft prompt for the frozen OPT.

SG-RQF adds the captions of retrieved training videos INSIDE the Q-Former:
  1. Retrieval: our CLIP4Clip finds the k=3 nearest training videos (scripts/build_rag_index.py).
  2. Consensus hints: from each neighbour we take its most central human caption (highest CIDEr
     against that video's other captions) instead of an arbitrary one.
  3. Hint tokens are appended after the 32 queries, so the queries self-attend to them while
     extracting visual features. The language model never sees the hint words directly.
  4. Similarity gate: every hint token carries a learned reliability embedding of its retrieval
     similarity (5 quantile buckets), and at test time hints below a similarity threshold
     (tuned on val) are masked out completely.
Training adds hint dropout (all hints removed for 20% of samples) so the model also works without them.
"""
import json, os, random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, Blip2Config, Blip2ForConditionalGeneration, Blip2Processor

from src.data.msvd import MSVDFrames
from src.models.blip2_video import TRAINABLE_PREFIXES, make_blip2_collate, prepare_blip2

SGRQF_PREFIXES = TRAINABLE_PREFIXES + ("hint_",)
N_QUERY = 32
N_BUCKETS = 5            # reliability buckets for hint similarity; index N_BUCKETS = [CLS]/[SEP]
BERT = "bert-base-uncased"   # the Q-Former was initialised from BERT-base: same vocabulary + width


# ----------------------------------------------------------------------------- model
def add_hint_modules(model, bert_name=BERT):
    """Hint embeddings (BERT word + position, like BLIP-2's Q-Former text input) + reliability embedding.
    The Q-Former's text feed-forward layers start as copies of its (pretrained) query feed-forward layers."""
    qf = model.qformer
    for layer in qf.encoder.layer:
        layer.intermediate.load_state_dict(layer.intermediate_query.state_dict())
        layer.output.load_state_dict(layer.output_query.state_dict())
    hidden = model.config.qformer_config.hidden_size
    bert = BertModel.from_pretrained(bert_name).embeddings
    assert bert.word_embeddings.embedding_dim == hidden, "BERT width must match the Q-Former width"
    model.hint_word_embeddings = nn.Embedding.from_pretrained(bert.word_embeddings.weight.detach().clone(), freeze=True)
    model.hint_position_embeddings = nn.Embedding.from_pretrained(bert.position_embeddings.weight.detach().clone(),
                                                                  freeze=False)
    model.hint_reliability = nn.Embedding(N_BUCKETS + 1, hidden)
    nn.init.zeros_(model.hint_reliability.weight)          # zero-init: starts as plain text hints
    for n, p in model.named_parameters():
        if n.startswith("qformer.") or n.startswith(("hint_position", "hint_reliability")):
            p.requires_grad = True
    return model


def load_sgrqf(base="Salesforce/blip2-opt-2.7b", num_frames=6, trainable_path=None, bert_name=BERT):
    processor = Blip2Processor.from_pretrained(base)
    config = Blip2Config.from_pretrained(base)
    config.qformer_config.use_qformer_text_input = True   # builds the Q-Former's text feed-forward layers
    try:
        model = Blip2ForConditionalGeneration.from_pretrained(base, config=config, dtype=torch.float16)
    except TypeError:  # older transformers
        model = Blip2ForConditionalGeneration.from_pretrained(base, config=config, torch_dtype=torch.float16)
    prepare_blip2(model, num_frames)
    add_hint_modules(model, bert_name)
    model.qformer.float()
    if trainable_path:
        sd = torch.load(trainable_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        print(f"loaded trained SG-RQF parts from {trainable_path} ({len(sd)} tensors)")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SG-RQF trainable parameters: {n_train / 1e6:.1f}M")
    return processor, BertTokenizerFast.from_pretrained(bert_name), model


def video_prefix_sgrqf(model, pixel_values, hint_ids=None, hint_mask=None, hint_rel=None):
    """(B, T, 3, H, W) + hint tokens (B, L) -> (B, T*32, lm_hidden). Same hints for every frame."""
    B, T = pixel_values.shape[:2]
    with torch.no_grad():
        vis = model.vision_model(pixel_values=pixel_values.flatten(0, 1).to(model.vision_model.dtype)).last_hidden_state
    q = model.query_tokens.expand(vis.shape[0], -1, -1)
    enc_mask = torch.ones(vis.shape[:2], dtype=torch.long, device=vis.device)
    if hint_ids is None:
        qo = model.qformer(query_embeds=q, encoder_hidden_states=vis, encoder_attention_mask=enc_mask).last_hidden_state
    else:
        L = hint_ids.shape[1]
        pos = torch.arange(L, device=hint_ids.device).unsqueeze(0)
        h = model.hint_word_embeddings(hint_ids) + model.hint_position_embeddings(pos) + model.hint_reliability(hint_rel)
        h = h.repeat_interleave(T, 0).to(q.dtype)                      # flatten(0,1) order = b0t0, b0t1, ...
        mask = torch.cat([torch.ones(q.shape[:2], dtype=torch.long, device=q.device),
                          hint_mask.repeat_interleave(T, 0).long()], 1)
        qo = model.qformer(query_embeds=torch.cat([q, h], 1), query_length=q.shape[1], attention_mask=mask,
                           encoder_hidden_states=vis, encoder_attention_mask=enc_mask).last_hidden_state[:, :q.shape[1]]
    qo = qo.view(B, T, qo.shape[1], -1) + model.frame_embedding[:T].unsqueeze(0).to(qo.dtype)
    return model.language_projection(qo.flatten(1, 2))


# ----------------------------------------------------------------------------- data
def load_hint_meta(hints_dir, split, k=3, select="consensus"):
    """{video: [(caption, similarity), ...]} from build_rag_index.py output (best neighbour first)."""
    meta = json.load(open(os.path.join(hints_dir, f"hints_{split}_meta.json")))
    key = "consensus_caption" if select == "consensus" else "caption"
    out = {}
    for v, rows in meta.items():
        out[v] = [(r[key] if r.get(key) else r["caption"], float(r["video_sim"])) for r in rows[:k]]
    return out


def similarity_edges(hints_dir, k=3):
    """Quantile edges of the TRAIN hint similarities -> reliability buckets (same edges at test time)."""
    sims = [s for rows in load_hint_meta(hints_dir, "train", k).values() for _, s in rows]
    return [float(x) for x in np.quantile(sims, [0.2, 0.4, 0.6, 0.8])]


class SGRQFFrames(MSVDFrames):
    """MSVDFrames + retrieved hints. gate: drop hints with similarity < gate (None = keep all)."""
    def __init__(self, root, split, processor, num_frames, train, repeats=1, hints_dir=None, k=3,
                 select="consensus", dropout=0.0, gate=None):
        super().__init__(root, split, processor, num_frames, train, repeats)
        self.hints = load_hint_meta(hints_dir, split, k, select)
        missing = [v for v in set(self.ids) if v not in self.hints]
        assert not missing, f"{len(missing)} {split} videos have no hints, e.g. {missing[:3]}"
        self.dropout, self.gate = dropout, gate

    def __getitem__(self, i):
        item = super().__getitem__(i)
        hints = self.hints[item["video_id"]]
        if self.gate is not None:
            hints = [h for h in hints if h[1] >= self.gate]
        if self.train and random.random() < self.dropout:
            hints = []
        item["hints"] = hints
        return item


def make_sgrqf_collate(lm_tokenizer, bert_tokenizer, edges, max_len=40, hint_len=24):
    """BLIP-2 caption collate + hint tensors: [CLS] hint1 [SEP] hint2 [SEP] hint3 [SEP]."""
    base = make_blip2_collate(lm_tokenizer, max_len)
    cls_id, sep_id = bert_tokenizer.cls_token_id, bert_tokenizer.sep_token_id
    edges = np.asarray(edges)

    def collate(batch):
        out = base(batch)
        seqs, rels = [], []
        for b in batch:
            ids, rel = [cls_id], [N_BUCKETS]
            for text, sim in b["hints"]:
                t = bert_tokenizer(text, add_special_tokens=False, truncation=True, max_length=hint_len)["input_ids"]
                bucket = int(np.searchsorted(edges, sim, side="right"))
                ids += t + [sep_id]; rel += [bucket] * len(t) + [N_BUCKETS]
            seqs.append(ids); rels.append(rel)
        L = max(len(s) for s in seqs)
        out["hint_ids"] = torch.tensor([s + [0] * (L - len(s)) for s in seqs])
        out["hint_rel"] = torch.tensor([r + [N_BUCKETS] * (L - len(r)) for r in rels])
        out["hint_mask"] = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        return out
    return collate


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_sgrqf(model, tokenizer, loader, device, num_beams=3, max_new_tokens=20):
    model.eval()
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
    bos = tokenizer.bos_token_id
    preds = {}
    for b in tqdm(loader, desc="generate", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            prefix = video_prefix_sgrqf(model, b["pixel_values"].to(device), b["hint_ids"].to(device),
                                        b["hint_mask"].to(device), b["hint_rel"].to(device))
            lm = model.language_model
            bos_emb = lm.get_input_embeddings()(torch.full((prefix.shape[0], 1), bos, device=device))
            inputs = torch.cat([prefix.to(bos_emb.dtype), bos_emb], 1)
            mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
            ids = lm.generate(inputs_embeds=inputs, attention_mask=mask, num_beams=num_beams,
                              max_new_tokens=max_new_tokens, eos_token_id=[nl, tokenizer.eos_token_id],
                              pad_token_id=tokenizer.pad_token_id)
        for vid, text in zip(b["video_ids"], tokenizer.batch_decode(ids, skip_special_tokens=True)):
            preds[vid] = text.strip()
    return preds
