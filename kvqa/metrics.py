"""Text-generation and atomic-attribute metrics with breakdowns."""
import re
import string

import numpy as np
import pandas as pd
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

_PUNCT = re.compile(f"[{re.escape(string.punctuation.replace('-', ''))}]")
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

ANSWER_TYPE = {  # coarse answer type of each atomic question class
    "text_presence": "yes/no", "box_artifact_presence": "yes/no", "finding_presence": "yes/no",
    "polyp_removal_status": "yes/no", "instrument_count": "count", "polyp_count": "count",
    "finding_count": "count", "abnormality_location": "location", "instrument_location": "location",
    "landmark_location": "location", "abnormality_color": "color", "landmark_color": "color",
    "abnormality_presence": "category", "polyp_type": "category", "landmark_presence": "category",
    "instrument_presence": "category", "procedure_type": "category", "polyp_size": "category",
}
TEXT_METRICS = ["exact_match", "bleu_sent", "rouge1", "rouge2", "rougeL", "meteor"]


def normalize(s):
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def per_sample(preds, refs):
    rows = []
    for p, r in zip(preds, refs):
        pn, rn = normalize(p), normalize(r)
        rg = _ROUGE.score(r, p)
        rows.append(dict(
            exact_match=float(pn == rn),
            bleu_sent=sacrebleu.sentence_bleu(pn, [rn]).score / 100,
            rouge1=rg["rouge1"].fmeasure, rouge2=rg["rouge2"].fmeasure, rougeL=rg["rougeL"].fmeasure,
            meteor=meteor_score([rn.split()], pn.split()) if pn else 0.0,
        ))
    return pd.DataFrame(rows)


def corpus_bleu(preds, refs):
    return sacrebleu.corpus_bleu([normalize(p) for p in preds], [[normalize(r) for r in refs]]).score / 100


def summarize(df, preds, refs):
    """df: per-row frame with metric columns + question_class(list), complexity, seen_image."""
    out = {"overall": {**df[TEXT_METRICS].mean().to_dict(), "bleu_corpus": corpus_bleu(preds, refs), "n": len(df)}}
    ex = df.assign(question_class=df.question_class.map(lambda l: sorted(set(l)))).explode("question_class")
    out["by_question_class"] = _group(ex, "question_class")
    out["by_complexity"] = _group(df, "complexity")
    out["by_seen_image"] = _group(df.assign(seen_image=df.seen_image.map({True: "seen", False: "unseen"})), "seen_image")
    if "seen_group" in df:  # unseen image AND no near-duplicate of it in training
        out["by_seen_group"] = _group(df.assign(seen_group=df.seen_group.map({True: "near-dup seen", False: "no near-dup"})), "seen_group")
    c1 = df[df.complexity == 1].assign(answer_type=lambda x: x.question_class.str[0].map(ANSWER_TYPE))
    out["by_answer_type_complexity1"] = _group(c1, "answer_type")
    return out


def _group(df, key):
    g = df.groupby(key)
    res = g[TEXT_METRICS].mean()
    res["n"] = g.size()
    return {str(k): v for k, v in res.round(4).to_dict(orient="index").items()}


def attribute_metrics(attr_enc, facts_by_img, logits_by_img, seen_imgs):
    """Image-level metrics of the atomic heads. logits_by_img: {img: {class: np.array}}."""
    rows = []
    for img, facts in facts_by_img.items():
        if img not in logits_by_img:
            continue
        enc = attr_enc.encode(facts)
        for c in attr_enc.classes:
            if c not in facts:
                continue
            lg = logits_by_img[img][c]
            t, _ = enc[c]
            if attr_enc.schema[c]["type"] == "multi":
                pred = (lg > 0).astype(np.float32)
                tp = (pred * t).sum()
                f1 = 1.0 if pred.sum() + t.sum() == 0 else 2 * tp / (pred.sum() + t.sum())
                rows.append(dict(cls=c, img=img, acc=float((pred == t).all()), f1=float(f1)))
            else:
                ok = float(t >= 0 and int(lg.argmax()) == t)
                rows.append(dict(cls=c, img=img, acc=ok, f1=ok))
    df = pd.DataFrame(rows)
    df["seen"] = df.img.isin(seen_imgs).map({True: "seen", False: "unseen"})
    res = {"overall": {"acc": df.acc.mean(), "f1": df.f1.mean(), "n": len(df)}}
    for key in ["cls", "seen"]:
        g = df.groupby(key)
        t = g[["acc", "f1"]].mean()
        t["n"] = g.size()
        res[f"by_{key}"] = t.round(4).to_dict(orient="index")
    return res


def bootstrap_ci(df, cols, key="img_id", reps=1000, seed=0):
    """95% CI of row-mean metrics, resampling whole images (rows of one image are correlated)."""
    g = df.groupby(key)[cols]
    sums, cnt = g.sum().to_numpy(), g.size().to_numpy()
    w = np.random.default_rng(seed).multinomial(len(cnt), np.full(len(cnt), 1 / len(cnt)), size=reps)
    means = (w @ sums) / (w @ cnt)[:, None]
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return {c: [float(lo[i]), float(hi[i])] for i, c in enumerate(cols)}
