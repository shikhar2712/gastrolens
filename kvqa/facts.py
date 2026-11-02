"""Rule-based clinical-fact check for single-fact (complexity-1) answers.

ROUGE/METEOR reward shared words, so "polyp measuring 5 to 10 millimeters" vs "... 11 to 20 millimeters" scores ~0.8.
Here each answer sentence is mapped to a canonical value of its atomic class (e.g. polyp_size -> "5-10mm") and
compared with the atomic ground truth from the `original` column.

Because reference sentences are LLM paraphrases that sometimes drop information ("scattered across multiple
regions"), a row is *checkable* only when extract(reference) == canonical(atomic label). Fact accuracy is then
extract(prediction) == canonical(atomic label) on checkable rows; an unparseable prediction counts as wrong.
Coverage (checkable / all) is reported per class so the metric's reach is explicit.

python -m kvqa.facts --pred experiments/<exp>/predictions/heldout_predictions.jsonl --cache data/cache_imgdisjoint
"""
import argparse, json, re

import pandas as pd

NUM = {"no": 0, "zero": 0, "none": 0, "one": 1, "single": 1, "a": 1, "an": 1, "solitary": 1, "two": 2, "three": 3,
       "four": 4, "five": 5, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
COLORS = {"pink": "pink", "pinkish": "pink", "red": "red", "reddish": "red", "white": "white", "whitish": "white",
          "yellow": "yellow", "yellowish": "yellow", "black": "black", "brown": "brown", "brownish": "brown",
          "grey": "grey", "gray": "grey", "greyish": "grey", "green": "green", "blue": "blue", "purple": "purple",
          "orange": "orange", "flesh": "flesh", "fleshy": "flesh"}
NEG = re.compile(r"^(no|not|none|without|absence|absent)\b|\bno (visible |identifiable |evident |significant |surgical |foreign |apparent )*"
                 r"(polyp|instrument|landmark|text|box|finding|abnormal|lesion|object|tool|device)")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9<>\-\s]", " ", s.lower()).split())


def negated(t):
    return bool(NEG.search(t))


def count(t, noun):
    """Number attached to the counted noun ('two polyps', 'no surgical instruments')."""
    if negated(t):
        return "0"
    m = re.search(rf"\b({'|'.join(NUM)})\b(?:\s+[a-z\-]+){{0,3}}\s+{noun}", t)
    return str(NUM[m.group(1)]) if m else None


def keywords(t, table, none_if_negated=True):
    if none_if_negated and negated(t):
        return ""
    found = sorted({v for k, v in table if re.search(k, t)})
    return ";".join(found) if found else None


def bands(t):
    """Locations, coarsened to the vertical bands mentioned (upper/middle/lower); fine grids are rarely verbalised."""
    if negated(t):
        return ""
    b = set()
    if re.search(r"\bupper|\btop|\bsuperior", t):
        b.add("upper")
    if re.search(r"\blower|\bbottom|\binferior", t):
        b.add("lower")
    # "upper-central" / "lower central" name a vertical band, not the middle; strip them before looking for the middle
    rest = re.sub(r"\b(upper|lower)[ -]cent(er|re|ral)\b", " ", t)
    if re.search(r"\bcent(er|re|ral)\b|\bmiddle|\bmid\b", rest):
        b.add("middle")
    return ";".join(sorted(b)) if b else None


def label_bands(lab):
    if lab in ("none", ""):
        return ""
    b = set()
    for r in lab.split(";"):
        r = r.strip().replace("rigth", "right")
        b.add("upper" if r.startswith("upper") else "lower" if r.startswith("lower") else "middle")
    return ";".join(sorted(b))


ABN = [(r"polyp", "polyp"), (r"colitis|inflammatory bowel", "ulcerative colitis"),
       (r"(o?esophag\w*|gastro-?o?esophageal) (inflammation|mucosal inflammation)|o?esophagitis", "oesophagitis"),
       (r"(?<!short-segment )(?<!short segment )barrett", "barretts"), (r"short[- ]segment barrett", "short-segment barretts")]
INSTR = [(r"forceps", "biopsy forceps"), (r"clip", "metal clip"), (r"snare", "polyp snare"), (r"\btube", "tube")]
LMK = [(r"c(a)?ec(um|al)", "cecum"), (r"ile(um|al)", "ileum"), (r"pylor", "pylorus"),
       (r"z[- ]line|gastro-?o?esophageal junction|squamocolumnar", "z-line")]
SIZE = [(r"(less than|under|below|smaller than|<)\s*5\b", "<5mm"), (r"\b5\s*(to|-)\s*10\b", "5-10mm"),
        (r"\b1[01]\s*(to|-)\s*20\b", "11-20mm"), (r"(greater than|larger than|more than|over|exceeding|above|>)\s*20\b", ">20mm")]
PTYPE = [(r"iia|flat|non-?polypoid|slightly elevated", "paris iia"), (r"\bip\b|pedunculated", "paris ip"),
         (r"\bis\b|sessile", "paris is")]


def extract(cls, text):
    """Canonical value of `cls` stated in `text`; None when the sentence does not pin it down."""
    t = norm(text)
    if cls == "text_presence":
        return "no" if negated(t) else "yes" if "text" in t else None
    if cls == "box_artifact_presence":
        return "no" if negated(t) else "yes" if "artifact" in t or "box" in t else None
    if cls == "procedure_type":
        c = bool(re.search(r"colon|lower g|large bowel|lower endoscop", t))
        g = bool(re.search(r"gastr|stomach|upper g|esophag|oesophag|upper endoscop|egd", t))
        return "colonoscopy" if c and not g else "gastroscopy" if g and not c else None
    if cls == "polyp_count":
        return count(t, "polyp")
    if cls == "instrument_count":
        return count(t, r"(surgical |endoscopic |medical |foreign )?(instrument|tool|device|object)")
    if cls == "finding_count":
        return None if negated(t) else count(t, r"(abnormal |distinct |notable |pathological |separate |discrete |significant )*(finding|abnormalit|lesion)")
    if cls == "abnormality_presence":
        return keywords(t, ABN)
    if cls == "instrument_presence":
        return keywords(t, INSTR)
    if cls == "landmark_presence":
        return keywords(t, LMK)
    if cls == "polyp_size":
        v = keywords(t, SIZE)
        return None if v and ";" in v else v
    if cls == "polyp_type":
        v = keywords(t, PTYPE)
        return None if v and ";" in v else v  # mixed morphology words ("pedunculated ... sessile features")
    if cls in ("abnormality_color", "landmark_color"):
        return keywords(t, [(rf"\b{k}\b", v) for k, v in COLORS.items()], none_if_negated=False)
    if cls in ("abnormality_location", "instrument_location", "landmark_location"):
        return bands(t)
    return None  # finding_presence (always "no"), polyp_removal_status (ambiguous phrasing)


def canonical(cls, lab):
    lab = lab.strip().lower()
    if cls.endswith("_location"):
        return label_bands(lab)
    if cls in ("abnormality_presence", "instrument_presence", "landmark_presence", "polyp_size", "polyp_type",
               "abnormality_color", "landmark_color"):
        v = ";".join(sorted({x.strip().replace(">20", ">20mm").replace("mmmm", "mm") for x in lab.split(";")} - {"none", ""}))
        return v
    return lab


def fact_table(df):
    """df: complexity-1 rows with columns question_class, original, answer, prediction."""
    rows = []
    for r in df.itertuples():
        cls = r.question_class[0]
        lab = canonical(cls, json.loads(r.original)[0]["a"])
        ref, pred = extract(cls, r.answer), extract(cls, r.prediction)
        rows.append(dict(img_id=r.img_id, cls=cls, checkable=ref is not None and ref == lab, correct=pred == lab,
                         parsed=pred is not None))
    return pd.DataFrame(rows)


def fact_summary(ft):
    ck = ft[ft.checkable]
    by = ck.groupby("cls").agg(fact_acc=("correct", "mean"), parsed=("parsed", "mean"), n=("correct", "size"))
    by["coverage"] = ft.groupby("cls").checkable.mean().reindex(by.index)
    return {"overall": {"fact_acc": float(ck.correct.mean()), "n": int(len(ck)), "coverage": float(ft.checkable.mean()),
                        "macro_fact_acc": float(by.fact_acc.mean())},
            "by_class": by.round(4).to_dict(orient="index")}


def with_original(pred_df, cache):
    sp = pd.read_parquet(f"{cache}/splits.parquet", columns=["orig_idx", "split", "img_id", "question", "original"])
    return pred_df.merge(sp, on=["orig_idx", "img_id", "question"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default=None, help="split name if orig_idx is ambiguous (train/heldout share it)")
    a = ap.parse_args()
    p = pd.read_json(a.pred, lines=True)
    m = with_original(p, a.cache)
    if a.split:
        m = m[m.split == a.split]
    c1 = m[m.complexity == 1]
    print(json.dumps(fact_summary(fact_table(c1)), indent=1))


if __name__ == "__main__":
    main()
