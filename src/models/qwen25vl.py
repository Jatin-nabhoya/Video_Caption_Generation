"""Qwen2.5-VL (3B-Instruct) for MSVD video captioning.
Frozen: vision encoder (fp16) + Qwen2.5 LLM (4-bit NF4, QLoRA).
Trained: LoRA adapters on the LLM attention + MLP layers only (~29M params).
Video: T frames -> Qwen's own video path (2 frames = 1 temporal patch, 2x2 spatial merge, MRoPE time ids).
Checkpoints contain ONLY the LoRA adapter (~115 MB) -> pushed to HF with src/utils/hub.py."""
import os, random, re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from src.data.msvd import MSVDFrames

PROMPT = "Describe this video in one short sentence."
# LoRA only on the LANGUAGE model: vision blocks are named "blocks.N", LLM blocks "layers.N"
LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def load_qwen25vl(base="Qwen/Qwen2.5-VL-3B-Instruct", adapter_path=None, lora_cfg=None, trainable=True, load_4bit=True):
    """Returns (processor, model). T4 has no bf16 -> everything runs in fp16."""
    from peft import LoraConfig, PeftModel, get_peft_model
    processor = AutoProcessor.from_pretrained(base)
    processor.tokenizer.padding_side = "left"          # needed for batched generation (also fine for training)
    kw = dict(device_map={"": 0}, attn_implementation="sdpa")
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["visual", "lm_head"])  # keep vision tower + output head in fp16
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype=torch.float16, **kw)
    except TypeError:  # older transformers
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16, **kw)
    model.config.use_cache = False

    if adapter_path:                                    # resume / evaluate a fine-tuned run
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
        print(f"loaded LoRA adapter from {adapter_path}")
    elif lora_cfg:                                      # fresh fine-tuning run
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return processor, model


class MSVDVideoFrames(MSVDFrames):
    """Same MSVD frames/splits as the other models, but returns raw PIL frames:
    Qwen's processor does its own resize/patching. Frames are resized to a fixed
    size (multiple of 28) so the number of visual tokens is constant."""
    def __init__(self, root, split, num_frames=8, frame_size=336, train=True, repeats=1):
        super().__init__(root, split, processor=None, num_frames=num_frames, train=train, repeats=repeats)
        assert num_frames % 2 == 0, "Qwen2.5-VL groups frames in pairs -> num_frames must be even"
        self.size = frame_size

    def __getitem__(self, i):
        vid = self.ids[i]
        frames = [f.resize((self.size, self.size), Image.BICUBIC) for f in self.load_frames(vid)]
        item = {"frames": frames, "video_id": vid}
        if self.train:
            item["caption"] = random.choice(self.caps[vid])
        return item


def _chat(processor, caption=None, prompt=PROMPT):
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    if caption is None:
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def make_qwen_collate(processor, prompt=PROMPT, fps=1.0):
    """Train batches: loss only on the caption tokens (+ <|im_end|>).
    Eval batches: prompt only (add_generation_prompt=True)."""
    tok = processor.tokenizer
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def collate(batch):
        train = "caption" in batch[0]
        texts = [_chat(processor, b["caption"].strip() if train else None, prompt) for b in batch]
        enc = processor(text=texts, videos=[b["frames"] for b in batch], padding=True,
                        return_tensors="pt", fps=fps)
        out = dict(enc)
        out["video_ids"] = [b["video_id"] for b in batch]
        if train:
            labels = torch.full_like(enc["input_ids"], -100)
            for r, ids in enumerate(enc["input_ids"].tolist()):
                start = max(i for i, t in enumerate(ids) if t == im_start) + 3   # skip "<|im_start|>assistant\n"
                end = max(i for i, t in enumerate(ids) if t == im_end) + 1       # keep <|im_end|> as the stop token
                labels[r, start:end] = enc["input_ids"][r, start:end]
            out["labels"] = labels
        return out
    return collate


def clean_caption(text):
    """MSVD references are short, lowercase, no final period. Keep the first sentence only."""
    t = text.strip().split("\n")[0]
    t = re.split(r"(?<=[.!?])\s", t)[0]
    t = re.sub(r"^(the|this) video (shows|depicts|features)\s+", "", t, flags=re.I)
    return t.strip().rstrip(".!").lower()


def model_inputs(batch, device):
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


@torch.no_grad()
def generate_qwen(model, processor, loader, device="cuda", num_beams=1, max_new_tokens=24):
    model.eval()
    tok = processor.tokenizer
    preds, raw = {}, {}
    for b in tqdm(loader, desc="generate", mininterval=30):
        inp = model_inputs(b, device)
        inp.pop("labels", None)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            ids = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for vid, t in zip(b["video_ids"], texts):
            raw[vid] = t.strip()
            preds[vid] = clean_caption(t)
    return preds, raw
