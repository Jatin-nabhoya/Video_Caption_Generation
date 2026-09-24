import os
import torch
from transformers import AutoProcessor, GitForCausalLM


def load_git(name_or_path="microsoft/git-base", num_frames=6):
    """Video GIT: one learnable temporal embedding per frame, like the GIT paper.

    When starting from the Hub base model, these embeddings don't exist in the checkpoint.
    Newer transformers versions leave them as UNINITIALIZED memory (huge random values),
    which breaks captions and gives NaN loss -> we set them to zero explicitly (paper init).
    Our own saved checkpoints (local folders) already contain trained values, so we keep those.
    """
    processor = AutoProcessor.from_pretrained(name_or_path)
    model = GitForCausalLM.from_pretrained(name_or_path, num_image_with_embedding=num_frames)
    if not os.path.isdir(name_or_path):
        with torch.no_grad():
            for p in model.git.img_temporal_embedding:
                p.zero_()
    emb_max = max(float(p.abs().max()) for p in model.git.img_temporal_embedding)
    print(f"temporal embeddings max |value| = {emb_max:.4f}")  # must be a small number, never 1e+20
    return processor, model