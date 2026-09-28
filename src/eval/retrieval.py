import numpy as np
import torch


def t2v_metrics(sim, gt):
    """sim: [num_captions, num_videos]; gt[i] = index of the correct video for caption i.
    Standard text-to-video retrieval metrics (as reported in the CLIP4Clip paper)."""
    correct = sim[torch.arange(len(gt), device=sim.device), gt].unsqueeze(1)
    ranks = ((sim > correct).sum(1) + 1).cpu().numpy()
    return {"R@1": round(float((ranks <= 1).mean() * 100), 2),
            "R@5": round(float((ranks <= 5).mean() * 100), 2),
            "R@10": round(float((ranks <= 10).mean() * 100), 2),
            "MdR": float(np.median(ranks)),
            "MnR": round(float(ranks.mean()), 1)}
