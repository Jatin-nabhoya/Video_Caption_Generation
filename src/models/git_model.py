import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, GitForCausalLM

ADAPTER_PREFIX = "git.temporal_adapter."


class TemporalAdapter(nn.Module):
    """Idea 2: let the frames talk to each other before the text decoder sees them.

    GIT encodes each frame on its own and only tags it with a frame number. This adapter takes the
    concatenated frame tokens (B, T*N, D), and lets every spatial token attend to the SAME position in
    the other T-1 frames (attention over time, as in TimeSformer's divided attention). Its output is added
    back through a gate that starts at 0, so at step 0 the model is exactly the baseline GIT.
    Only the adapter's CHANGE (encoder output minus its input) is gated - gating the whole output
    (v1) mostly rescaled the original features, so the gate never learned.
    """
    def __init__(self, dim, num_frames, layers=1, heads=8, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.time_pos = nn.Parameter(torch.zeros(num_frames, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, L, D = x.shape
        T = self.num_frames
        if L % T:                       # single image or unexpected shape: leave untouched
            return x
        N = L // T
        z = x.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)   # one short sequence per spatial token
        z = z + self.time_pos.to(z.dtype)
        delta = self.encoder(z) - z     # ONLY what the adapter adds (the encoder layer keeps a residual copy of z)
        delta = delta.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B, L, D)
        return x + torch.tanh(self.gate).to(x.dtype) * delta


def add_temporal_adapter(model, num_frames, adapter_cfg, checkpoint_dir=None):
    """Attach the adapter in front of GIT's visual projection (via a forward pre-hook, so the original
    weights keep their names). Loads trained adapter weights from checkpoint_dir if they are there."""
    dim = model.config.vision_config.hidden_size
    model.git.temporal_adapter = TemporalAdapter(dim, num_frames, adapter_cfg.get("layers", 1),
                                                 adapter_cfg.get("heads", 8), adapter_cfg.get("dropout", 0.1))
    model.git.temporal_adapter.train(model.training)   # match the model's mode (no dropout when evaluating)
    model.git.visual_projection.register_forward_pre_hook(lambda mod, args: (model.git.temporal_adapter(args[0]),))
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        from safetensors.torch import load_file
        sd = {}
        for f in sorted(os.listdir(checkpoint_dir)):
            if f.endswith(".safetensors"):
                sd.update({k[len(ADAPTER_PREFIX):]: v for k, v in load_file(os.path.join(checkpoint_dir, f)).items()
                           if k.startswith(ADAPTER_PREFIX)})
        if sd:
            model.git.temporal_adapter.load_state_dict(sd)
            print(f"temporal adapter: loaded {len(sd)} tensors, gate = {torch.tanh(model.git.temporal_adapter.gate.detach()).item():.4f}")
        else:
            print("temporal adapter: no trained weights in checkpoint -> fresh (gate = 0, identical to baseline)")
    n = sum(p.numel() for p in model.git.temporal_adapter.parameters())
    print(f"temporal adapter: {adapter_cfg.get('layers', 1)} layer(s), {n / 1e6:.2f}M parameters")
    return model


def load_git(name_or_path="microsoft/git-base", num_frames=6, temporal_adapter=None):
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
    emb_max = max(float(p.detach().abs().max()) for p in model.git.img_temporal_embedding)
    print(f"temporal embeddings max |value| = {emb_max:.4f}")  # must be a small number, never 1e+20
    if temporal_adapter:
        add_temporal_adapter(model, num_frames, temporal_adapter, name_or_path)
    return processor, model

def caption_loss(logits, labels):
    """Next-word loss on the caption tokens only.
    Written out explicitly because some transformers versions shift the labels TWICE inside GIT,
    which silently trains the model to predict the word 2 positions ahead (scrambled captions)."""
    import torch.nn.functional as F
    n_text = labels.shape[1]
    text_logits = logits[:, -n_text:-1, :].float()   # image tokens come first, caption tokens last
    target = labels[:, 1:]                             # token t predicts token t+1
    return F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)), target.reshape(-1), ignore_index=-100)
