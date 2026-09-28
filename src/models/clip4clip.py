"""CLIP4Clip (meanP) - Luo et al. 2021.
Every frame is encoded by CLIP's image encoder, frame embeddings are averaged ("mean pooling")
into one video embedding, and videos/captions are matched with a contrastive loss.
CLIP4Clip has NO caption decoder, so we caption a video by RETRIEVING the best-matching
caption from the training-set captions (retrieval-based captioning)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_clip4clip(name_or_path="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(name_or_path)
    model = CLIPModel.from_pretrained(name_or_path)
    return processor, model


def _emb(out):
    # transformers v4 returns a tensor, v5 returns an output object with .pooler_output
    return out if torch.is_tensor(out) else out.pooler_output


def encode_videos(model, pixel_values):
    """pixel_values: (B, T, 3, H, W) -> (B, D) normalized video embeddings (meanP)."""
    B, T = pixel_values.shape[:2]
    f = _emb(model.get_image_features(pixel_values=pixel_values.flatten(0, 1)))
    f = F.normalize(f.float(), dim=-1).view(B, T, -1).mean(1)
    return F.normalize(f, dim=-1)


def encode_texts(model, input_ids, attention_mask):
    t = _emb(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
    return F.normalize(t.float(), dim=-1)


def contrastive_loss(v, t, logit_scale):
    """Symmetric InfoNCE: video i should match caption i (and vice versa) within the batch."""
    logits = logit_scale.exp().clamp(max=100) * v @ t.t()
    labels = torch.arange(len(v), device=v.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


@torch.no_grad()
def encode_text_list(model, tokenizer, texts, device, bs=256, max_len=32):
    out = []
    for i in range(0, len(texts), bs):
        tok = tokenizer(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out.append(encode_texts(model, tok["input_ids"].to(device), tok["attention_mask"].to(device)))
    return torch.cat(out)


@torch.no_grad()
def encode_video_set(model, loader, device):
    ids, embs = [], []
    for b in tqdm(loader, desc="encode videos", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            embs.append(encode_videos(model, b["pixel_values"].to(device)))
        ids += b["video_ids"]
    return ids, torch.cat(embs)
