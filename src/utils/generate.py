import torch
from tqdm import tqdm

@torch.no_grad()
def generate_captions(model, processor, loader, device, num_beams=3, max_length=30):
    model.eval()
    preds = {}
    for batch in tqdm(loader, desc="generate", mininterval=30):
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(pixel_values=batch["pixel_values"].to(device), max_length=max_length, num_beams=num_beams,
                                 use_cache=False)  # cache breaks when image tokens > 1024
        for vid, text in zip(batch["video_ids"], processor.batch_decode(ids, skip_special_tokens=True)):
            preds[vid] = text.strip()
    return preds
