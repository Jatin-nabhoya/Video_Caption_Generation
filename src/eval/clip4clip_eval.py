import torch
from src.eval.metrics import score_captions
from src.eval.retrieval import t2v_metrics
from src.models.clip4clip import encode_text_list, encode_video_set


def build_caption_pool(train_caps):
    """All unique training captions = the 'answer bank' CLIP4Clip retrieves from."""
    return sorted({c for caps in train_caps.values() for c in caps})


@torch.no_grad()
def evaluate_clip4clip(model, tokenizer, ds, loader, device, pool_texts):
    model.eval()
    vids, V = encode_video_set(model, loader, device)
    # 1) retrieval: every caption of the split is a query, ranked against all videos of the split
    texts, gt = [], []
    for j, v in enumerate(vids):
        for c in ds.caps[v]:
            texts.append(c); gt.append(j)
    T = encode_text_list(model, tokenizer, texts, device)
    retrieval = t2v_metrics(T @ V.t(), torch.tensor(gt, device=device))
    # 2) captioning by retrieval: each video gets the closest TRAINING caption
    P = encode_text_list(model, tokenizer, pool_texts, device)
    best = (V @ P.t()).argmax(1).tolist()
    preds = {v: pool_texts[i] for v, i in zip(vids, best)}
    scores = score_captions(preds, ds.caps)
    return scores, retrieval, preds
