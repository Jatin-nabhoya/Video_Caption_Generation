"""BLIP-2 (OPT-2.7B) for video captioning.
Frozen: ViT-g image encoder + OPT-2.7B language model (fp16).
Trained: Q-Former + query tokens + language projection + a new per-frame embedding (fp32).
Video: each of T frames -> 32 Q-Former query outputs (+ frame-position embedding) -> T*32 soft tokens for OPT.
Checkpoints contain ONLY the trained parts (~0.45 GB instead of ~8 GB)."""
import json, os
import torch
import torch.nn as nn
from huggingface_hub import HfApi, create_repo, snapshot_download
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

TRAINABLE_PREFIXES = ("qformer.", "query_tokens", "language_projection.", "frame_embedding")


def prepare_blip2(model, num_frames):
    """Add the frame embedding, freeze ViT + OPT, keep trainable parts in fp32."""
    hidden = model.config.qformer_config.hidden_size
    model.frame_embedding = nn.Parameter(torch.zeros(num_frames, 1, hidden))  # zero-init = plain concatenation at start
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(TRAINABLE_PREFIXES)
    model.qformer.float(); model.language_projection.float()
    model.query_tokens.data = model.query_tokens.data.float()
    return model


def load_blip2(base="Salesforce/blip2-opt-2.7b", num_frames=6, trainable_path=None):
    processor = Blip2Processor.from_pretrained(base)
    try:
        model = Blip2ForConditionalGeneration.from_pretrained(base, dtype=torch.float16)
    except TypeError:  # older transformers
        model = Blip2ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16)
    prepare_blip2(model, num_frames)
    if trainable_path:
        sd = torch.load(trainable_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        print(f"loaded trained BLIP-2 parts from {trainable_path} ({len(sd)} tensors)")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BLIP-2 trainable parameters: {n_train / 1e6:.1f}M")
    return processor, model


def video_prefix(model, pixel_values):
    """(B, T, 3, H, W) -> (B, T*num_query, lm_hidden) soft tokens for the language model."""
    B, T = pixel_values.shape[:2]
    with torch.no_grad():
        vis = model.vision_model(pixel_values=pixel_values.flatten(0, 1).to(model.vision_model.dtype)).last_hidden_state
    q = model.query_tokens.expand(vis.shape[0], -1, -1)
    qo = model.qformer(query_embeds=q, encoder_hidden_states=vis,
                       encoder_attention_mask=torch.ones(vis.shape[:2], dtype=torch.long, device=vis.device)).last_hidden_state
    qo = qo.view(B, T, qo.shape[1], -1) + model.frame_embedding[:T].unsqueeze(0).to(qo.dtype)
    return model.language_projection(qo.flatten(1, 2))


def lm_logits(model, prefix, input_ids, attention_mask):
    emb = model.language_model.get_input_embeddings()(input_ids)
    inputs = torch.cat([prefix.to(emb.dtype), emb], dim=1)
    mask = torch.cat([torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], 1)
    return model.language_model(inputs_embeds=inputs, attention_mask=mask).logits


def make_blip2_collate(tokenizer, max_len=40):
    """Captions end with a newline: BLIP-2/OPT uses '\\n' as its end-of-caption token."""
    def collate(batch):
        out = {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
               "video_ids": [b["video_id"] for b in batch]}
        if "caption" in batch[0]:
            tok = tokenizer([b["caption"].strip() + "\n" for b in batch], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            labels = tok["input_ids"].clone()
            labels[tok["attention_mask"] == 0] = -100
            out.update(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], labels=labels)
        return out
    return collate


@torch.no_grad()
def generate_blip2(model, tokenizer, loader, device, num_beams=3, max_new_tokens=20):
    model.eval()
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
    bos = tokenizer.bos_token_id
    preds = {}
    for b in tqdm(loader, desc="generate", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            prefix = video_prefix(model, b["pixel_values"].to(device))
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


def push_trainable(model, state, repo_id, subfolder, local_root="/kaggle/working/ckpt"):
    path = os.path.join(local_root, subfolder); os.makedirs(path, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(TRAINABLE_PREFIXES)}
    torch.save(sd, os.path.join(path, "trainable.pt"))
    json.dump(state, open(os.path.join(path, "state.json"), "w"), indent=2)
    create_repo(repo_id, private=True, exist_ok=True)
    HfApi().upload_folder(folder_path=path, repo_id=repo_id, path_in_repo=subfolder,
                          commit_message=f"{subfolder} epoch {state.get('epoch')}")
    print(f"pushed -> {repo_id}/{subfolder}")


def try_resume_trainable(repo_id, subfolder):
    try:
        root = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
        path = os.path.join(root, subfolder)
        return os.path.join(path, "trainable.pt"), json.load(open(os.path.join(path, "state.json")))
    except Exception as e:
        print(f"no checkpoint at {repo_id}/{subfolder} ({type(e).__name__}) -> starting fresh")
        return None, None
