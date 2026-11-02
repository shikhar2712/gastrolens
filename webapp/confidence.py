"""How sure the model is: the probabilities behind the fact classifier's decisions.

The page's headline is the classifier's abnormality decision and its polyp line is the classifier's polyp count and size, so the
confidence shown is the probability the classifier itself gives to that decision: a sigmoid for the multi-label heads (abnormality,
size) and a softmax for the single-label ones (polyp count). It is the model's own number, not a measured chance of being right:
it is not calibrated, and on images the model was trained on it is close to 100%.
"""
import torch


@torch.no_grad()
def head_probs(pr, image_path):
    """{fact class: {label: probability}} for one image, or None for a predictor without a model (the test stub)."""
    if not hasattr(pr, "model"):
        return None
    m, dev = pr.model, pr.device
    img = pr.load_image(image_path).unsqueeze(0).float().div(255).sub(0.5).div(0.5).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
        _, _, logits = m.encode_images(img)
    out = {}
    for c, lg in logits.items():
        s, v = m.attr_schema[c], lg[0].float()
        p = torch.sigmoid(v) if s["type"] == "multi" else torch.softmax(v, -1)
        out[c] = dict(zip(s["labels"], p.tolist()))
    return out


def pct(p):
    """A probability as a whole percent, rounded down and never 100: a network's probability is not a certainty (the page shows 99 as
    "99%+"), and a decision that was made is never 0."""
    return None if p is None else max(1, min(99, int(p * 100)))


def summarize(probs, facts):
    """{'abnormality': %, 'polyp': %, 'polyp_size': %} (only what applies) from head_probs and the facts ({fact key: value})."""
    if not probs:
        return {}
    out = {}
    abn = (facts.get("abnormality_presence") or "").split(";")[0].strip()
    if abn and abn in probs.get("abnormality_presence", {}):
        out["abnormality"] = pct(probs["abnormality_presence"][abn])
    count = probs.get("polyp_count", {})
    if count and str(facts.get("polyp_count") or "0").strip() not in ("", "0"):
        out["polyp"] = pct(1.0 - count.get("0", 0.0))  # the probability that there is at least one polyp
        size = str(facts.get("polyp_size") or "").split(";")[0].strip()
        if size and size in probs.get("polyp_size", {}):
            out["polyp_size"] = pct(probs["polyp_size"][size])
    return out
