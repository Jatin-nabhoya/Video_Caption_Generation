import types
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _cache_frame_features(model):
    """GIT re-encodes every frame with the vision encoder at EVERY generated word.
    Frames don't change while one batch is captioned, so encode them once and reuse."""
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
    return restore


def _generate(model, pix, num_beams, max_length):
    """Generate captions; if the GPU runs out of memory, split the batch in half and retry."""
    oom = False
    restore = _cache_frame_features(model)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            return model.generate(pixel_values=pix, max_length=max_length, num_beams=num_beams,
                                  use_cache=False)  # cache breaks when image tokens > 1024
    except torch.OutOfMemoryError:
        oom = True
    finally:
        restore()
    if oom:
        if pix.shape[0] == 1:
            raise RuntimeError("Out of GPU memory even for 1 video - lower num_beams or num_frames")
        torch.cuda.empty_cache()
        half = pix.shape[0] // 2
        a = _generate(model, pix[:half], num_beams, max_length)
        b = _generate(model, pix[half:], num_beams, max_length)
        L = max(a.shape[1], b.shape[1])
        pad = model.config.pad_token_id or 0
        a = F.pad(a, (0, L - a.shape[1]), value=pad)
        b = F.pad(b, (0, L - b.shape[1]), value=pad)
        return torch.cat([a, b])


@torch.no_grad()
def generate_captions(model, processor, loader, device, num_beams=3, max_length=20):
    model.eval()
    preds = {}
    for batch in tqdm(loader, desc="generate", mininterval=30):
        ids = _generate(model, batch["pixel_values"].to(device), num_beams, max_length)
        for vid, text in zip(batch["video_ids"], processor.batch_decode(ids, skip_special_tokens=True)):
            preds[vid] = text.strip()
    return preds
