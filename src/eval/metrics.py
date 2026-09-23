from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def score_captions(preds, refs):
    """Return BLEU-4, METEOR, ROUGE-L, and CIDEr scores multiplied by 100."""
    tok = PTBTokenizer()
    gts = tok.tokenize({k: [{"caption": c} for c in refs[k]] for k in preds})
    res = tok.tokenize({k: [{"caption": preds[k]}] for k in preds})
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    meteor, _ = Meteor().compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    raw = {"BLEU-4": bleu[3], "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider}
    return {k: round(float(v) * 100, 2) for k, v in raw.items()}
