"""Section 3 - Experimental analysis for MSVD video captioning.

Compares CLIP4Clip, GIT, BLIP-2 and Qwen2.5-VL (zero-shot and fine-tuned) on the
MSVD test split and writes every table and figure needed for the presentation:

  01 overall metrics (4 models x 2 settings)
  02 per-class metrics (MSVD has no classes, so videos are bucketed by content)
  03 error taxonomy: WHY a caption failed (object error, action error,
     hallucination, gender confusion, generic, verbose, counting)
  04 difficulty factors: which clip properties make videos hard
  05 worst-case analysis (hard for every model) with reasons
  06 best-case analysis (easy for every model) with reasons
  07 good / bad / random examples per class (image | GT | model 1..4)
  08 model-specific wins and failures
  09 strengths & weaknesses summary (auto-drafted, edit before presenting)

Run on Kaggle (CPU is enough; METEOR and the PTB tokenizer need Java, which Kaggle has):
  python scripts/experiment_analysis.py --data_root /kaggle/input/.../msvd
"""
import argparse, glob, json, os, random, re, textwrap
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------------------------
# Models and prediction files (all live in results/)
# ----------------------------------------------------------------------------
MODELS = {
    "CLIP4Clip": ("clip4clip_zeroshot_test_preds.json", "clip4clip_meanP_v1_best_test_preds.json"),
    "GIT": ("git_base_zeroshot_test_preds.json", "git_base_ft_v2_best_test_preds.json"),
    "BLIP-2": ("blip2_zeroshot_test_preds.json", "blip2_v1_best_test_preds.json"),
    "Qwen2.5-VL": ("qwen25vl_zeroshot_test_preds.json", "qwen25vl_3b_lora_v1_best_test_preds.json"),
}
SETTINGS = {"ZS": "Zero-shot", "FT": "Fine-tuned"}
MODEL_COLORS = {"CLIP4Clip": "#4C78A8", "GIT": "#F58518", "BLIP-2": "#54A24B", "Qwen2.5-VL": "#B279A2"}
METRICS = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]

# ----------------------------------------------------------------------------
# Lexicons for content classes and error analysis
# ----------------------------------------------------------------------------
W = lambda s: set(s.split())
CLASS_KEYWORDS = {
    "Animals": W("dog dogs puppy puppies cat cats kitten kittens kitty bird birds parrot horse horses cow cows "
                 "monkey monkeys elephant elephants panda pandas lion tiger fish hamster rabbit rabbits bunny "
                 "squirrel mouse mice rat duck ducks hen goat pig pigs sheep bear bears deer turtle frog snake "
                 "animal animals giraffe zebra penguin penguins owl kangaroo otter otters sloth lemur hedgehog "
                 "chimpanzee gorilla camel dolphin seal shark slow loris"),
    "Cooking & Food": W("cooking cook cooks cutting cuts slicing slices chopping chops peeling peels frying fries "
                        "boiling mixing mixes stirring stirs pouring pours onion onions potato potatoes tomato "
                        "tomatoes meat chicken egg eggs bowl pan pot kitchen food vegetables vegetable oil butter "
                        "flour dough bread cheese garlic carrot carrots shrimp sauce rice noodles cake recipe "
                        "ingredients spices salt pepper seasoning dish baking bakes grating cucumber lemon fruit "
                        "eating eats"),
    "Vehicles": W("car cars truck trucks bus motorcycle motorcycles bike bicycle driving drives drove train "
                  "plane airplane aeroplane jet boat boats ship helicopter vehicle jeep scooter tractor "
                  "motorbike racing"),
    "Music & Dance": W("guitar piano drums drum violin flute singing sings sing song songs music musical band "
                       "keyboard instrument trumpet saxophone harmonica cello dancing dances dance dancer "
                       "dancers performing"),
    "Sports & Exercise": W("ball soccer football basketball tennis cricket baseball golf swimming swims running "
                           "runs jumping jumps exercise exercising exercises lifting weights gym boxing kicking "
                           "skating skateboard skateboarding skiing surfing wrestling fighting martial cycling "
                           "volleyball goal shooting hoop pushups push ups workout trampoline"),
}
PERSON = W("man men woman women boy boys girl girls person people lady ladies guy guys child children kid kids "
           "baby babies toddler player players chef someone gentleman teenager toddlers")
MALE = W("man men boy boys guy guys gentleman he his")
FEMALE = W("woman women girl girls lady ladies she her")
PLURAL_SUBJ = W("men women boys girls people ladies guys children kids babies players two three four several "
                "group some dogs cats puppies kittens")
ANIMALS = CLASS_KEYWORDS["Animals"] - W("animal animals")
OBJECT_NOUNS = ANIMALS | CLASS_KEYWORDS["Vehicles"] | W("ball guitar piano drums violin flute keyboard knife "
                                                       "bowl pan pot phone computer camera toy book paper")
SYNONYMS = [W("man guy gentleman person someone he"), W("woman lady person someone she"),
            W("boy kid child"), W("girl kid child"), W("dog puppy"), W("cat kitten kitty"),
            W("talking speaking saying telling"), W("slicing cutting chopping"), W("bike bicycle cycle"),
            W("motorcycle motorbike bike"), W("car vehicle"), W("plane airplane aeroplane jet"),
            W("sofa couch"), W("putting placing adding"), W("mixing stirring"), W("running jogging"),
            W("riding driving"), W("baby infant"), W("frying cooking"), W("playing performing")]
STOP = W("a an the is are was were be been being of in on at to for with and or but from by into onto over "
         "under up down out off it its this that these those there their his her he she they them some "
         "very while as near next front back top his has have having do does doing being just also then "
         "something thing things s")
GENERIC = W("doing something thing things stuff showing shows video clip someone person people")
HYPERNYMS = W("animal animals creature creatures pet pets person people someone thing object vehicle")

def norm(w):
    """Very light normalisation so 'dogs'~'dog'. Good enough for a lexical audit."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def words(text):
    return re.findall(r"[a-z]+", text.lower())

def content(text):
    return [w for w in words(text) if w not in STOP and len(w) > 2]

def syn_expand(ws):
    out = set(ws)
    for w in list(ws):
        for group in SYNONYMS:
            if w in group:
                out |= group
    return out | {norm(w) for w in out}

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def find_data_root(arg):
    cands = [arg] if arg else []
    cands += ["/kaggle/input/datasets/jatinnabhoya/video-caption-generation/msvd",
              "/kaggle/input/video-caption-generation/msvd", "data/msvd"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "msvd_annotations.json")):
            return c
    hits = glob.glob("/kaggle/input/**/msvd_annotations.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("msvd_annotations.json not found - pass --data_root")

def load_preds(results_dir):
    preds = {}
    for model, files in MODELS.items():
        for setting, fname in zip(SETTINGS, files):
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                print(f"  [missing] {path} - skipping {model} {setting}")
                continue
            preds[(model, setting)] = {k: str(v).strip() for k, v in json.load(open(path)).items()}
    return preds

def duration_of(vid):
    try:
        _, s, e = vid.rsplit("_", 2)
        return max(float(e) - float(s), 0.0)
    except ValueError:
        return np.nan

# ----------------------------------------------------------------------------
# Scoring: per-video BLEU-4 / METEOR / ROUGE-L / CIDEr with pycocoevalcap
# ----------------------------------------------------------------------------
def score_all(preds, refs, use_meteor=True):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    vids = sorted(refs)
    gts = tok.tokenize({v: [{"caption": c} for c in refs[v]] for v in vids})
    meteor = None
    if use_meteor:
        from pycocoevalcap.meteor.meteor import Meteor
        meteor = Meteor()
    per_video, overall, tokenized = {}, {}, {}
    for key, p in preds.items():
        missing = [v for v in vids if v not in p]
        if missing:
            print(f"  {key}: {len(missing)} test videos have no prediction - scored as empty")
        res = tok.tokenize({v: [{"caption": p.get(v, "") or "empty"}] for v in vids})
        tokenized[key] = res
        b, b_each = Bleu(4).compute_score(gts, res, verbose=0)
        r, r_each = Rouge().compute_score(gts, res)
        c, c_each = Cider().compute_score(gts, res)
        if meteor:
            m, m_each = meteor.compute_score(gts, res)
        else:
            m, m_each = np.nan, [np.nan] * len(vids)
        overall[key] = {"BLEU-4": b[3] * 100, "METEOR": m * 100, "ROUGE-L": r * 100, "CIDEr": c * 100}
        per_video[key] = pd.DataFrame({"video": vids, "BLEU-4": np.array(b_each[3]) * 100,
                                       "METEOR": np.array(m_each, dtype=float) * 100,
                                       "ROUGE-L": np.array(r_each) * 100, "CIDEr": np.array(c_each) * 100})
        print(f"  scored {key[0]:<11} {key[1]}  CIDEr={c*100:6.2f}")
    return per_video, overall, gts, tokenized

def corpus_bleu4(gts, res, vids):
    from pycocoevalcap.bleu.bleu import Bleu
    b, _ = Bleu(4).compute_score({v: gts[v] for v in vids}, {v: res[v] for v in vids}, verbose=0)
    return b[3] * 100

# ----------------------------------------------------------------------------
# Video-level descriptors: class, consensus, rarity, brightness, motion
# ----------------------------------------------------------------------------
def assign_class(caps):
    n = len(caps)
    hits = {c: sum(any(norm(w) in {norm(k) for k in kws} or w in kws for w in words(cap)) for cap in caps) / n
            for c, kws in CLASS_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    if hits[best] >= 0.25:
        return best
    person = sum(any(w in PERSON for w in words(cap)) for cap in caps) / n
    return "People & Daily life" if person >= 0.25 else "Other"

def ref_consensus(caps, max_refs=25):
    sets = [set(map(norm, content(c))) for c in caps[:max_refs]]
    sims = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    return float(np.mean(sims)) if sims else np.nan

def train_idf(train_caps):
    df = Counter()
    for caps in train_caps.values():
        for c in caps:
            df.update(set(map(norm, content(c))))
    n = sum(len(c) for c in train_caps.values())
    return {w: np.log(n / (1 + d)) for w, d in df.items()}, np.log(n)

def concept_rarity(caps, idf, max_idf):
    cnt = Counter(w for c in caps for w in set(map(norm, content(c))))
    core = [w for w, k in cnt.items() if k / len(caps) >= 0.3] or [w for w, _ in cnt.most_common(3)]
    return float(np.mean([idf.get(w, max_idf) for w in core]))

def frame_files(data_root, vid):
    return sorted(glob.glob(os.path.join(data_root, "frames", vid, "*.jpg")))

def frame_stats(data_root, vid, n=6):
    files = frame_files(data_root, vid)
    if not files:
        return np.nan, np.nan
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int)
    arr = [np.asarray(Image.open(files[i]).convert("L").resize((64, 64)), dtype=np.float32) for i in idx]
    bright = float(np.mean([a.mean() for a in arr]))
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(arr[:-1], arr[1:])])) if len(arr) > 1 else 0.0
    return bright, motion

# ----------------------------------------------------------------------------
# Error taxonomy: why did a caption fail?
# ----------------------------------------------------------------------------
ERROR_ORDER = ["Object / subject error", "Gender / age confusion", "Action error", "Hallucinated details",
               "Verbose / style mismatch", "Too generic", "Counting error", "Missed main event",
               "Wording only (meaning OK)"]
ERROR_COLORS = ["#E45756", "#FF9DA6", "#F58518", "#B279A2", "#72B7B2", "#9D755D", "#EECA3B", "#4C78A8", "#BAB0AC"]

def subjects(text):
    ws = words(text)
    subj = [norm(w) for w in ws if w in PERSON or w in ANIMALS or w in CLASS_KEYWORDS["Vehicles"]]
    return subj[:1]

def gender(text):
    ws = words(text)[:6]
    if any(w in MALE for w in ws):
        return "M"
    if any(w in FEMALE for w in ws):
        return "F"
    return None

def verbs(text):
    return {w for w in words(text) if w.endswith("ing") and len(w) > 4}

def is_plural(text):
    return any(w in PLURAL_SUBJ for w in words(text)[:4])

def diagnose(pred, caps):
    """Return the list of error flags for one prediction against its references."""
    flags = []
    ref_words = syn_expand({w for c in caps for w in words(c)})
    ref_subj = Counter(s for c in caps for s in subjects(c))
    p_subj = subjects(pred)
    vague_subject = bool(set(words(pred)[:5]) & HYPERNYMS) and not p_subj
    if vague_subject:
        flags.append("Too generic")
    elif p_subj:
        s = p_subj[0]
        sub_ok = syn_expand({s}) & set(ref_subj)
        person_both = s in {norm(w) for w in PERSON} and any(r in {norm(w) for w in PERSON} for r in ref_subj)
        if not sub_ok and not person_both:
            flags.append("Object / subject error")
    elif ref_subj and not (set(map(norm, content(pred))) & set(ref_subj)):
        flags.append("Object / subject error")
    g_pred = gender(pred)
    g_refs = Counter(g for g in (gender(c) for c in caps) if g)
    if g_pred and g_refs and g_refs.get(g_pred, 0) / sum(g_refs.values()) < 0.3:
        flags.append("Gender / age confusion")
    pv = verbs(pred)
    rv = syn_expand({v for c in caps for v in verbs(c)})
    if pv and not (syn_expand(pv) & rv):
        flags.append("Action error")
    unsupported = [w for w in content(pred) if w not in ref_words and norm(w) not in ref_words]
    # short captions: one invented object ("a man is playing a flute") is already a hallucination
    if len(unsupported) >= 2 or (unsupported and len(content(pred)) <= 3 and
                                 any(w not in PERSON and not w.endswith("ing") for w in unsupported)):
        flags.append("Hallucinated details")
    ref_len = np.median([len(words(c)) for c in caps])
    if len(words(pred)) > max(1.8 * ref_len, 14):
        flags.append("Verbose / style mismatch")
    pc = content(pred)
    if (len(pc) <= 1 or (set(pc) & GENERIC and len(pc) <= 3)) and "Too generic" not in flags:
        flags.append("Too generic")
    ref_plural = np.mean([is_plural(c) for c in caps])
    if is_plural(pred) != (ref_plural >= 0.5) and abs(ref_plural - 0.5) > 0.3:
        flags.append("Counting error")
    return flags, unsupported

def primary_error(flags, cider=100.0):
    for e in ERROR_ORDER:
        if e in flags:
            return e
    # no rule fired: near-zero CIDEr means the caption is plausible but describes the wrong event
    return "Wording only (meaning OK)" if cider >= 20 else "Missed main event"

# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

def montage(data_root, vid, n=3, h=120):
    files = frame_files(data_root, vid)
    if not files:
        return None
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    ims = [Image.open(files[i]).convert("RGB") for i in idx]
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    out = Image.new("RGB", (sum(im.width for im in ims) + 4 * (n - 1), h), "white")
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + 4
    return out

def cider_color(c):
    return "#1a7f37" if c >= 100 else ("#9a6700" if c >= 40 else "#cf222e")

def example_grid(rows, models, data_root, title, path, wrap=26):
    """rows: list of dicts with video, refs, tag, preds{model:(text,cider,error)}"""
    if not rows:
        return
    ncol = 2 + len(models)
    fig = plt.figure(figsize=(3.3 * ncol, 2.15 * len(rows) + 0.8))
    gs = fig.add_gridspec(len(rows) + 1, ncol, height_ratios=[0.18] + [1] * len(rows),
                          width_ratios=[1.6, 1.25] + [1] * len(models), hspace=0.25, wspace=0.08,
                          top=1 - 0.55 / fig.get_figheight(), bottom=0.01, left=0.01, right=0.99)
    heads = ["Clip (3 frames)", "Ground truth (2 of the refs)"] + [f"{m}" for m in models]
    for j, h in enumerate(heads):
        ax = fig.add_subplot(gs[0, j]); ax.axis("off")
        ax.text(0.5, 0.2, h, ha="center", va="center", fontsize=11, weight="bold",
                color=MODEL_COLORS.get(h, "black"))
    for i, r in enumerate(rows, start=1):
        ax = fig.add_subplot(gs[i, 0]); ax.axis("off")
        img = montage(data_root, r["video"])
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "(frames not found)", ha="center", va="center", color="gray")
        ax.set_title(f"{r.get('tag', '')}  {r['video']}  [{r.get('cls', '')}]", fontsize=7.5, loc="left")
        ax = fig.add_subplot(gs[i, 1]); ax.axis("off")
        gt = "\n".join(textwrap.fill("- " + c, wrap) for c in r["refs"][:2])
        ax.text(0, 0.95, gt, va="top", fontsize=8.5, family="DejaVu Sans")
        for j, m in enumerate(models, start=2):
            ax = fig.add_subplot(gs[i, j]); ax.axis("off")
            text, c, err = r["preds"].get(m, ("-", 0, ""))
            ax.text(0, 0.95, textwrap.fill(text, wrap), va="top", fontsize=8.5)
            note = f"CIDEr {c:.0f}" + (f"\n{err}" if err else "")
            ax.text(0, 0.02, note, va="bottom", fontsize=7.5, color=cider_color(c), weight="bold")
    fig.suptitle(title, fontsize=13, weight="bold", y=1 - 0.25 / fig.get_figheight())
    save(fig, path)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def df_to_md(df, floatfmt=".2f"):
    try:
        return df.to_markdown(floatfmt=floatfmt)
    except ImportError:
        return "```\n" + df.round(2).to_string() + "\n```"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--setting", default="FT", choices=["FT", "ZS"], help="setting used for the example picks")
    ap.add_argument("--k", type=int, default=8, help="rows in the worst/best-case figures")
    ap.add_argument("--no_meteor", action="store_true")
    ap.add_argument("--no_frames", action="store_true", help="skip brightness/motion and thumbnails")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_class", type=int, default=10, help="smallest class shown in per-class results")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)

    data_root = find_data_root(a.data_root)
    ann = json.load(open(os.path.join(data_root, "msvd_annotations.json")))
    refs = ann["test"]
    print(f"data_root={data_root}  test videos={len(refs)}")
    preds = load_preds(a.results_dir)
    models = [m for m in MODELS if (m, "FT") in preds or (m, "ZS") in preds]
    S = a.setting

    # ---- per-video scores ------------------------------------------------
    print("[1/9] scoring every prediction per video")
    per_video, overall, gts, tokenized = score_all(preds, refs, use_meteor=not a.no_meteor)

    # ---- video descriptors -----------------------------------------------
    print("[2/9] describing videos (class, consensus, rarity, frames)")
    idf, max_idf = train_idf(ann.get("train", {}))
    vrows = []
    for v, caps in refs.items():
        b, mo = (np.nan, np.nan) if a.no_frames else frame_stats(data_root, v)
        vrows.append({"video": v, "class": assign_class(caps), "duration_s": duration_of(v),
                      "ref_consensus": ref_consensus(caps), "concept_rarity": concept_rarity(caps, idf, max_idf),
                      "brightness": b, "motion": mo, "n_refs": len(caps),
                      "ref_len": float(np.median([len(words(c)) for c in caps]))})
    vdf = pd.DataFrame(vrows).set_index("video")
    vdf["duration_bin"] = pd.cut(vdf["duration_s"], [-1, 5, 10, 1e9], labels=["short (<=5s)", "medium (5-10s)", "long (>10s)"])
    train_caps = {c.strip().lower() for caps in ann.get("train", {}).values() for c in caps}

    # ---- long table: one row per (video, model, setting) -----------------
    long_rows = []
    for (m, s), pv in per_video.items():
        for row in pv.itertuples(index=False):
            v = row.video
            pred = preds[(m, s)].get(v, "")
            flags, unsup = diagnose(pred, refs[v])
            long_rows.append({"video": v, "model": m, "setting": s, "pred": pred,
                              "BLEU-4": row[1], "METEOR": row[2], "ROUGE-L": row[3], "CIDEr": row[4],
                              "flags": "; ".join(flags), "unsupported_words": " ".join(unsup),
                              "n_unsupported": len(unsup), "pred_len": len(words(pred)),
                              "exact_ref_match": pred.strip().lower() in {c.strip().lower() for c in refs[v]},
                              "copied_train_caption": pred.strip().lower() in train_caps})
    L = pd.DataFrame(long_rows).join(vdf, on="video")
    # failure = bottom quartile of CIDEr for that model+setting, or CIDEr < 30
    q25 = L.groupby(["model", "setting"])["CIDEr"].transform(lambda x: x.quantile(0.25))
    L["is_failure"] = (L["CIDEr"] <= q25) | (L["CIDEr"] < 30)
    L["primary_error"] = [primary_error(f.split("; ") if f else [], c) if fail else ""
                          for f, fail, c in zip(L["flags"], L["is_failure"], L["CIDEr"])]
    L.to_csv(os.path.join(a.out, "per_video_scores.csv"), index=False)
    vdf.to_csv(os.path.join(a.out, "video_descriptors.csv"))

    # ---- 01 overall ------------------------------------------------------
    print("[3/9] overall table")
    ov = pd.DataFrame([{"Model": m, "Setting": SETTINGS[s], **{k: v for k, v in overall[(m, s)].items()}}
                       for m in models for s in SETTINGS if (m, s) in overall]).set_index(["Model", "Setting"])
    ov.to_csv(os.path.join(a.out, "01_overall.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, met in zip(axes, METRICS):
        for i, m in enumerate(models):
            for j, s in enumerate(SETTINGS):
                if (m, s) in overall:
                    ax.bar(i + (j - 0.5) * 0.38, overall[(m, s)][met], 0.36, color=MODEL_COLORS[m],
                           alpha=0.45 if s == "ZS" else 1, hatch="//" if s == "ZS" else None, edgecolor="white")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=9); ax.set_title(met)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Overall MSVD test scores (hatched = zero-shot, solid = fine-tuned)", weight="bold")
    save(fig, os.path.join(fig_dir, "01_overall.png"))

    # ---- 02 per class ----------------------------------------------------
    print("[4/9] per-class table")
    counts = vdf["class"].value_counts()
    # classes with fewer than --min_class videos give noisy numbers; they still count in ALL
    classes = [c for c in list(CLASS_KEYWORDS) + ["People & Daily life", "Other"] if counts.get(c, 0) >= a.min_class]
    dropped = [c for c in counts.index if c not in classes]
    if dropped:
        print(f"  classes left out of per-class tables (< {a.min_class} videos): {dropped}")
    pc_rows = []
    for (m, s) in per_video:
        for c in classes + ["ALL"]:
            vids = list(vdf.index) if c == "ALL" else list(vdf.index[vdf["class"] == c])
            sub = L[(L.model == m) & (L.setting == s) & (L.video.isin(vids))]
            pc_rows.append({"Model": m, "Setting": SETTINGS[s], "Class": c, "n_videos": len(vids),
                            "BLEU-4": corpus_bleu4(gts, tokenized[(m, s)], vids),
                            "METEOR": sub["METEOR"].mean(), "ROUGE-L": sub["ROUGE-L"].mean(),
                            "CIDEr": sub["CIDEr"].mean(), "failure_rate_%": 100 * sub["is_failure"].mean()})
    PC = pd.DataFrame(pc_rows)
    PC.to_csv(os.path.join(a.out, "02_per_class.csv"), index=False)
    for s in SETTINGS:
        heat = PC[(PC.Setting == SETTINGS[s])].pivot(index="Model", columns="Class", values="CIDEr")
        heat = heat.reindex(index=[m for m in models if m in heat.index], columns=classes + ["ALL"])
        if heat.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.6 * heat.shape[1] + 2, 0.7 * heat.shape[0] + 1.6))
        im = ax.imshow(heat.values, cmap="RdYlGn", vmin=0, vmax=max(200, np.nanmax(heat.values)))
        ax.set_xticks(range(heat.shape[1])); ax.set_yticks(range(heat.shape[0]))
        n_per = vdf["class"].value_counts()
        ax.set_xticklabels([f"{c.replace(' & ', ' &' + chr(10))}\n(n={n_per.get(c, len(vdf))})" for c in heat.columns], fontsize=8.5)
        ax.set_yticklabels(heat.index)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center", fontsize=10, weight="bold")
        ax.set_title(f"Per-class CIDEr - {SETTINGS[s]}", weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.025)
        save(fig, os.path.join(fig_dir, f"02_per_class_cider_{s}.png"))
    gain = PC.pivot_table(index=["Model", "Class"], columns="Setting", values="CIDEr")
    if {"Zero-shot", "Fine-tuned"} <= set(gain.columns):
        gain["FT gain"] = gain["Fine-tuned"] - gain["Zero-shot"]
        gain.to_csv(os.path.join(a.out, "02_per_class_finetune_gain.csv"))
    # per duration bin (bonus axis)
    dur = L.groupby(["model", "setting", "duration_bin"], observed=True)["CIDEr"].mean().unstack()
    dur.to_csv(os.path.join(a.out, "02b_per_duration_cider.csv"))

    # ---- 03 error taxonomy -----------------------------------------------
    print("[5/9] error taxonomy")
    F = L[L.is_failure]
    tax = (F.groupby(["model", "setting"]) ["primary_error"].value_counts(normalize=True).mul(100)
           .unstack(fill_value=0).reindex(columns=ERROR_ORDER, fill_value=0))
    tax.to_csv(os.path.join(a.out, "03_error_taxonomy_pct_of_failures.csv"))
    # flag rates over ALL predictions (not only failures)
    flag_rate = pd.DataFrame({e: L.groupby(["model", "setting"])["flags"].apply(lambda x: 100 * x.str.contains(re.escape(e)).mean())
                              for e in ERROR_ORDER[:-1]})
    style = L.groupby(["model", "setting"]).agg(mean_len=("pred_len", "mean"),
                                                unsupported_words_per_cap=("n_unsupported", "mean"),
                                                exact_ref_match_pct=("exact_ref_match", lambda x: 100 * x.mean()),
                                                copied_train_caption_pct=("copied_train_caption", lambda x: 100 * x.mean()),
                                                failure_rate_pct=("is_failure", lambda x: 100 * x.mean()))
    style["vocab_size"] = L.groupby(["model", "setting"]) ["pred"].apply(lambda x: len({w for p in x for w in words(p)}))
    pd.concat([style, flag_rate.add_suffix(" (% of all)")], axis=1).to_csv(os.path.join(a.out, "03_caption_behaviour.csv"))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(tax) + 1.5))
    left = np.zeros(len(tax))
    labels = [f"{m} ({SETTINGS[s]})" for m, s in tax.index]
    for e, col in zip(ERROR_ORDER, ERROR_COLORS):
        ax.barh(labels, tax[e].values, left=left, color=col, label=e, edgecolor="white")
        for y, (l, w) in enumerate(zip(left, tax[e].values)):
            if w >= 7:
                ax.text(l + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8, color="white", weight="bold")
        left += tax[e].values
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% of failed captions (primary reason)")
    ax.legend(ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)
    ax.set_title("Why captions fail - error taxonomy", weight="bold"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, os.path.join(fig_dir, "03_error_taxonomy.png"))
    tax_cls = (F[F.setting == S].groupby(["class"]) ["primary_error"].value_counts(normalize=True).mul(100)
               .unstack(fill_value=0).reindex(columns=ERROR_ORDER, fill_value=0))
    tax_cls.to_csv(os.path.join(a.out, f"03b_error_taxonomy_by_class_{S}.csv"))

    # ---- 04 difficulty factors ------------------------------------------
    print("[6/9] difficulty factors")
    from scipy.stats import spearmanr
    factors = ["ref_consensus", "concept_rarity", "duration_s", "brightness", "motion", "ref_len"]
    diff_rows = []
    for (m, s) in per_video:
        sub = L[(L.model == m) & (L.setting == s)]
        for f in factors:
            ok = sub[f].notna()
            if ok.sum() > 10 and sub.loc[ok, f].nunique() > 1:
                rho, p = spearmanr(sub.loc[ok, f], sub.loc[ok, "CIDEr"])
                diff_rows.append({"Model": m, "Setting": SETTINGS[s], "Factor": f, "spearman_rho": rho, "p_value": p})
    D = pd.DataFrame(diff_rows)
    D.to_csv(os.path.join(a.out, "04_difficulty_factors.csv"), index=False)
    good_f = [f for f in factors if vdf[f].notna().sum() > 10 and vdf[f].nunique() > 3]
    if good_f:
        fig, axes = plt.subplots(1, len(good_f), figsize=(3.4 * len(good_f), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, good_f):
            bins = pd.qcut(vdf[f].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
            for m in models:
                sub = L[(L.model == m) & (L.setting == S)].copy()
                sub["bin"] = sub["video"].map(bins)
                y = sub.groupby("bin", observed=True)["CIDEr"].mean()
                ax.plot(range(len(y)), y.values, marker="o", color=MODEL_COLORS[m], label=m)
            ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"], fontsize=8)
            ax.set_title(f.replace("_", " "), fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(f"mean CIDEr ({SETTINGS[S]})"); axes[-1].legend(fontsize=8)
        fig.suptitle("What makes a clip hard? CIDEr by quartile of each clip property", weight="bold", y=1.04)
        save(fig, os.path.join(fig_dir, "04_difficulty_factors.png"))

    # ---- helper to build example rows -----------------------------------
    T = L[L.setting == S]
    piv = T.pivot(index="video", columns="model", values="CIDEr").reindex(columns=models)
    mean_c = piv.mean(axis=1)
    def row_for(v, tag=""):
        r = {"video": v, "refs": refs[v], "tag": tag, "cls": vdf.loc[v, "class"], "preds": {}}
        for m in models:
            x = T[(T.video == v) & (T.model == m)]
            if len(x):
                x = x.iloc[0]
                r["preds"][m] = (x["pred"], x["CIDEr"], x["primary_error"] if x["is_failure"] else "")
        return r
    def why_hard(v):
        d = vdf.loc[v]; out = []
        if d.ref_consensus < vdf.ref_consensus.quantile(0.25): out.append("annotators disagree (low reference consensus)")
        if d.concept_rarity > vdf.concept_rarity.quantile(0.75): out.append("rare concept in training captions")
        if pd.notna(d.brightness) and d.brightness < vdf.brightness.quantile(0.15): out.append("dark footage")
        if pd.notna(d.motion) and d.motion > vdf.motion.quantile(0.85): out.append("fast motion / camera shake")
        if pd.notna(d.duration_s) and d.duration_s <= 3: out.append("very short clip")
        if pd.notna(d.duration_s) and d.duration_s >= 20: out.append("long clip, several events")
        return out or ["no clip-level red flag - model-side errors"]
    def why_easy(v):
        d = vdf.loc[v]; out = []
        if d.ref_consensus > vdf.ref_consensus.quantile(0.75): out.append("annotators agree (high consensus)")
        if d.concept_rarity < vdf.concept_rarity.quantile(0.25): out.append("very common concept in training data")
        if d["class"] in ("Animals", "Vehicles", "Music & Dance", "Sports & Exercise"): out.append(f"distinctive {d['class'].lower()} content")
        if pd.notna(d.motion) and d.motion < vdf.motion.quantile(0.5): out.append("stable camera, single clear action")
        return out or ["simple single-subject scene"]
    def case_md(rows, header, reason_fn):
        lines = [f"# {header} ({SETTINGS[S]} models)\n"]
        for i, r in enumerate(rows, 1):
            v = r["video"]
            lines.append(f"## {i}. `{v}` - class: {r['cls']} - mean CIDEr {mean_c[v]:.1f}")
            lines.append("**Reference captions (first 3):** " + " | ".join(refs[v][:3]))
            lines.append("**Clip-level reasons:** " + "; ".join(reason_fn(v)) + "\n")
            lines.append("| Model | Prediction | CIDEr | Diagnosed problems | Unsupported words |")
            lines.append("|---|---|---:|---|---|")
            for m in models:
                x = T[(T.video == v) & (T.model == m)]
                if len(x):
                    x = x.iloc[0]
                    lines.append(f"| {m} | {x['pred']} | {x['CIDEr']:.1f} | {x['flags'] or '-'} | {x['unsupported_words'] or '-'} |")
            lines.append("")
        return "\n".join(lines)

    # ---- 05 / 06 worst and best cases -----------------------------------
    print("[7/9] worst-case and best-case analysis")
    worst = list(mean_c.sort_values().index[:a.k])
    best = list(mean_c.sort_values(ascending=False).index[:a.k])
    data_for_img = None if a.no_frames else data_root
    example_grid([row_for(v, "WORST") for v in worst], models, data_for_img or "",
                 f"Worst cases - clips every model gets wrong ({SETTINGS[S]})", os.path.join(fig_dir, "05_worst_cases.png"))
    example_grid([row_for(v, "BEST") for v in best], models, data_for_img or "",
                 f"Best cases - clips every model gets right ({SETTINGS[S]})", os.path.join(fig_dir, "06_best_cases.png"))
    open(os.path.join(a.out, "05_worst_cases.md"), "w").write(case_md([row_for(v) for v in worst], "Worst-case analysis", why_hard))
    open(os.path.join(a.out, "06_best_cases.md"), "w").write(case_md([row_for(v) for v in best], "Best-case analysis", why_easy))
    # profile of the hardest / easiest 10% vs all videos
    n10 = max(1, len(mean_c) // 10)
    prof = pd.DataFrame({"worst 10%": vdf.loc[mean_c.sort_values().index[:n10], factors].mean(),
                         "all videos": vdf[factors].mean(),
                         "best 10%": vdf.loc[mean_c.sort_values().index[-n10:], factors].mean()})
    cls_prof = pd.DataFrame({"worst 10% (%)": vdf.loc[mean_c.sort_values().index[:n10], "class"].value_counts(normalize=True) * 100,
                             "all (%)": vdf["class"].value_counts(normalize=True) * 100,
                             "best 10% (%)": vdf.loc[mean_c.sort_values().index[-n10:], "class"].value_counts(normalize=True) * 100}).fillna(0)
    prof.to_csv(os.path.join(a.out, "05_06_profile_factors.csv")); cls_prof.to_csv(os.path.join(a.out, "05_06_profile_classes.csv"))

    # ---- 07 per-class good / bad / random -------------------------------
    print("[8/9] per-class good / bad / random examples")
    for c in classes:
        vids = [v for v in vdf.index[vdf["class"] == c] if v in mean_c.index]
        if not vids:
            continue
        s_ = mean_c[vids].sort_values()
        picks = [(s_.index[-1], "GOOD"), (s_.index[0], "BAD")]
        rest = [v for v in vids if v not in (s_.index[-1], s_.index[0])]
        if rest:
            picks.append((rng.choice(rest), "RANDOM"))
        safe = re.sub(r"[^A-Za-z]+", "_", c).strip("_").lower()
        example_grid([row_for(v, t) for v, t in picks], models, data_for_img or "",
                     f"{c}: good / bad / random ({SETTINGS[S]})", os.path.join(fig_dir, f"07_class_{safe}.png"))
    # per model x class good/bad/random table (for the appendix / backup slides)
    gbr = []
    for m in models:
        for c in classes:
            sub = T[(T.model == m) & (T["class"] == c)].sort_values("CIDEr")
            if sub.empty:
                continue
            for tag, x in [("good", sub.iloc[-1]), ("bad", sub.iloc[0]), ("random", sub.sample(1, random_state=a.seed).iloc[0])]:
                gbr.append({"model": m, "class": c, "pick": tag, "video": x.video, "pred": x.pred, "CIDEr": round(x.CIDEr, 1),
                            "reference": refs[x.video][0], "problems": x.flags})
    pd.DataFrame(gbr).to_csv(os.path.join(a.out, "07_good_bad_random_per_model_class.csv"), index=False)

    # ---- 08 model-specific wins and failures ----------------------------
    spec_rows, spec_md = [], ["# Model-specific wins and failures\n"]
    for m in models:
        others = piv.drop(columns=[m]).mean(axis=1)
        delta = piv[m] - others
        for v, tag in [(delta.idxmin(), f"{m} FAILS"), (delta.idxmax(), f"{m} WINS")]:
            spec_rows.append(row_for(v, tag))
            spec_md.append(f"- **{tag}** `{v}`: {m} CIDEr {piv.loc[v, m]:.1f} vs others {others[v]:.1f}. "
                           f"{m}: \"{preds[(m, S)].get(v, '')}\" | ref: \"{refs[v][0]}\"")
    example_grid(spec_rows, models, data_for_img or "", f"Model-specific wins and failures ({SETTINGS[S]})",
                 os.path.join(fig_dir, "08_model_specific.png"))
    open(os.path.join(a.out, "08_model_specific.md"), "w").write("\n".join(spec_md) + "\n")

    # ---- 09 strengths and weaknesses ------------------------------------
    print("[9/9] strengths & weaknesses summary")
    sw = ["# Experimental strengths & weaknesses (auto-drafted - check and edit)\n"]
    for m in models:
        sub = PC[(PC.Model == m) & (PC.Setting == SETTINGS[S]) & (PC.Class != "ALL")].sort_values("CIDEr")
        if sub.empty:
            continue
        ftz = PC[(PC.Model == m) & (PC.Class == "ALL")].set_index("Setting")["CIDEr"]
        errs = tax.loc[(m, S)] if (m, S) in tax.index else pd.Series(dtype=float)
        st = style.loc[(m, S)]
        sw.append(f"## {m}")
        sw.append(f"- Overall CIDEr: zero-shot {ftz.get('Zero-shot', np.nan):.1f} -> fine-tuned {ftz.get('Fine-tuned', np.nan):.1f}")
        sw.append(f"- **Strongest classes:** {', '.join(f'{r.Class} ({r.CIDEr:.0f})' for r in sub.iloc[::-1][:2].itertuples())}")
        sw.append(f"- **Weakest classes:** {', '.join(f'{r.Class} ({r.CIDEr:.0f})' for r in sub[:2].itertuples())}")
        if len(errs):
            top = errs.sort_values(ascending=False)[:2]
            sw.append(f"- **Main failure reasons:** {', '.join(f'{k} ({v:.0f}% of failures)' for k, v in top.items())}")
        sw.append(f"- Caption style: {st.mean_len:.1f} words on average, {st.unsupported_words_per_cap:.2f} unsupported words per caption, "
                  f"vocabulary {int(st.vocab_size)} words, {st.copied_train_caption_pct:.0f}% copied verbatim from a training caption")
        md = D[(D.Model == m) & (D.Setting == SETTINGS[S])].copy()
        if len(md):
            md = md.reindex(md.spearman_rho.abs().sort_values(ascending=False).index)
            r0 = md.iloc[0]
            sw.append(f"- Most sensitive to **{r0.Factor.replace('_', ' ')}** (Spearman rho = {r0.spearman_rho:+.2f})")
        sw.append("")
    open(os.path.join(a.out, "09_strengths_weaknesses.md"), "w").write("\n".join(sw))

    # ---- index -----------------------------------------------------------
    rep = ["# Section 3 - Experimental analysis\n", f"Example picks use the **{SETTINGS[S]}** models. "
           "Classes are keyword buckets built from the reference captions (MSVD has no labels).\n",
           "## Overall\n", df_to_md(ov), "\n## Class sizes\n", df_to_md(vdf["class"].value_counts().to_frame("videos")),
           f"\n## Per-class CIDEr ({SETTINGS[S]})\n",
           df_to_md(PC[PC.Setting == SETTINGS[S]].pivot(index="Class", columns="Model", values="CIDEr")),
           "\n## Error taxonomy (% of failed captions)\n", df_to_md(tax, ".0f"),
           "\n## Caption behaviour\n", df_to_md(style),
           "\n## Difficulty factors (Spearman rho with CIDEr)\n",
           df_to_md(D[D.Setting == SETTINGS[S]].pivot(index="Factor", columns="Model", values="spearman_rho")),
           "\n## Worst 10% vs best 10% of clips\n", df_to_md(prof), "\n", df_to_md(cls_prof, ".0f")]
    open(os.path.join(a.out, "README.md"), "w").write("\n".join(rep) + "\n")
    print(f"\nDone. Everything is in {a.out}/ (start with README.md and figures/)")

if __name__ == "__main__":
    main()
