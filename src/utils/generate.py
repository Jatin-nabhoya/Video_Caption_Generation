import types
import torch
from tqdm import tqdm


def _cache_frame_features(model):
    """GIT re-encodes every frame with the vision encoder at EVERY generated word.
    Frames don't change while one batch is captioned, so encode them once and reuse.
    Returns (store, restore)."""
    enc = model.git.image_encoder
    original_forward = enc.forward
    store = {}

    def cached_forward(pixel_values, *args, **kwargs):
        key = (pixel_values.data_ptr(), tuple(pixel_values.shape), tuple(pixel_values.stride()))
        if key not in store:
            store[key] = original_forward(pixel_values, *args, **kwargs).last_hidden_state
        # clone: GIT adds the temporal embedding IN-PLACE to this tensor
        return types.SimpleNamespace(last_hidden_state=store[key].clone())

    enc.forward = cached_forward

    def restore():
        store.clear()
        enc.forward = original_forward
    return store, restore


@torch.no_grad()
def generate_captions(model, processor, loader, device, num_beams=3, max_length=20):
    model.eval()
    preds = {}
    for batch in tqdm(loader, desc="generate", mininterval=30):
        store, restore = _cache_frame_features(model)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                ids = model.generate(pixel_values=batch["pixel_values"].to(device),
                                     max_length=max_length, num_beams=num_beams,
                                     use_cache=False)  # cache breaks when image tokens > 1024
        finally:
            restore()
        for vid, text in zip(batch["video_ids"], processor.batch_decode(ids, skip_special_tokens=True)):
            preds[vid] = text.strip()
    return preds
