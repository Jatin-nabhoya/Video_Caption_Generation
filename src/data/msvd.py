import glob, json, os, random
import torch
from PIL import Image
from torch.utils.data import Dataset

class MSVDFrames(Dataset):
    """Reads pre-extracted frames + msvd_annotations.json."""
    def __init__(self, root, split, processor, num_frames=6, train=True, repeats=1):
        self.caps = json.load(open(os.path.join(root, "msvd_annotations.json")))[split]
        self.root, self.proc, self.nf, self.train = root, processor, num_frames, train
        vids = sorted(self.caps)
        self.ids = [v for v in vids for _ in range(repeats)] if train else vids

    def __len__(self):
        return len(self.ids)

    def load_frames(self, vid):
        files = sorted(glob.glob(os.path.join(self.root, "frames", vid, "*.jpg")))
        idx = torch.linspace(0, len(files) - 1, self.nf).round().long().tolist()
        return [Image.open(files[i]).convert("RGB") for i in idx]

    def __getitem__(self, i):
        vid = self.ids[i]
        pix = self.proc(images=self.load_frames(vid), return_tensors="pt").pixel_values
        item = {"pixel_values": pix, "video_id": vid}
        if self.train:
            item["caption"] = random.choice(self.caps[vid])
        return item

def make_collate(tokenizer, max_len=40):
    def collate(batch):
        out = {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "video_ids": [b["video_id"] for b in batch]}
        if "caption" in batch[0]:
            tok = tokenizer([b["caption"] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            labels = tok.input_ids.clone()
            labels[tok.attention_mask == 0] = -100
            out.update(input_ids=tok.input_ids, attention_mask=tok.attention_mask, labels=labels)
        return out
    return collate
