"""The demo's suggested questions, as a tree, and the rule that opens a branch.

The page lists the parent questions for the chosen image. A follow-up row appears under a parent only after that parent has been
asked and the model's answer says the thing is there: "Are there any polyps visible in the image?" -> "one polyp identified" opens
size, type and count. "No polyps observed" keeps them closed, because the model answers a follow-up about something that is not
there with a confident number anyway (a "polyp measuring 11 to 20 millimeters" on an image it just said has no polyp). Each
parent has at most MAX_FOLLOW_UPS follow-ups. The gate is the model's own answer, not the 17 facts, which disagree with the
answers on some images.

Every question here is a question the model was trained on (tests/test_questions.py checks them against the training set), so the
router sends all of them to the model and the answers are in the distribution it was trained on.
"""
import re

OVERVIEW = "What procedure is depicted in the image?"  # asked on every image: its answer and Grad-CAM head the report

BRANCHES = [
    {"id": "procedure", "question": OVERVIEW, "children": []},
    {"id": "abnormality", "question": "Are there any abnormalities visible in the image?", "children": [
        "Where is the abnormality located in the image?",
        "What colors are observed in the abnormal areas?",
        "How many abnormalities are visible?"]},
    {"id": "polyp", "question": "Are there any polyps visible in the image?", "children": [
        "What is the size of the polyp?",
        "What type of polyp is present?",
        "How many polyps are visible in the image?"]},
    {"id": "instrument", "question": "Are there any surgical instruments visible in the image?", "children": [
        "How many surgical instruments are visible in the image?",
        "In which regions of the image is the instrument located?"]},
    {"id": "landmark", "question": "Which anatomical landmarks are visible in the image?", "children": [
        "Which regions of the image contain the anatomical landmark?"]},
    {"id": "text", "question": "Is there any text visible on the image?", "children": []},
    {"id": "box", "question": "Are there any green or black box artifacts present?", "children": []},
]
MAX_FOLLOW_UPS = 3

# The page no longer asks questions. Its analysis is the model's standard reading of an image: these five questions are put to it
# in one batch, and the answers are the findings (in this order, as the model gave them, none filtered). The Grad-CAM shown is the
# one for the abnormality answer: it gave a real finding on all seven sample images.
STANDARD = [b["question"] for b in BRANCHES if b["id"] in ("procedure", "abnormality", "polyp", "instrument", "landmark")]
CAM_QUESTION = next(b["question"] for b in BRANCHES if b["id"] == "abnormality")
POLYP_QUESTION = next(b["question"] for b in BRANCHES if b["id"] == "polyp")
INSTRUMENT_QUESTION = next(b["question"] for b in BRANCHES if b["id"] == "instrument")

# The 17 facts, in related groups; any fact not listed goes under "Other".
FACT_GROUPS = [
    ["Abnormality", ["abnormality_presence", "abnormality_location", "abnormality_color"]],
    ["Findings and polyps", ["finding_presence", "finding_count", "polyp_count", "polyp_size", "polyp_type", "polyp_removal_status"]],
    ["Instruments", ["instrument_presence", "instrument_count", "instrument_location"]],
    ["Landmarks", ["landmark_presence", "landmark_location", "landmark_color"]],
    ["Image and procedure", ["procedure_type", "text_presence", "box_artifact_presence"]],
]

# The model's negative answers: "No polyps observed", "No surgical instruments identified", "No anatomical landmarks identified",
# "No residual polyp tissue identified"; and the same said another way.
_NEGATIVE = re.compile(
    r"^\W*(no|none|nothing|not|without|absence|negative)\b"
    r"|\b(absent|not (present|visible|seen|detected|identified|observed|found)|no (evidence|sign|signs|visible))\b",
    re.I)


def is_positive(answer):
    """Whether an answer says that what was asked about is there ("one polyp identified", "tube visible in the gastric region",
    "evidence of z-line appearance"). An empty answer is not positive."""
    a = (answer or "").strip()
    return bool(a) and not _NEGATIVE.search(a)


def tree():
    """The tree for the client: [{id, question, children}]."""
    return [{"id": b["id"], "question": b["question"], "children": list(b["children"])} for b in BRANCHES]


# The model's fact classifier is more reliable than its generated answer for whether there is a polyp: on 153 held-out images with a
# labelled polyp count it was right on presence for all 153 (the generated "polyps visible?" answer: 96%, missing 12% of the images that
# had one). Its size band is a weaker signal: exactly right in 73% of 26 labelled cases, so it is shown as a model prediction.
_SIZES = {"<5mm": "less than 5 mm", "5-10mm": "5 to 10 mm", "11-20mm": "11 to 20 mm", ">20mm": "more than 20 mm"}
_COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def polyp_detail(facts):
    """"One polyp identified, 5 to 10 mm" from the fact classifier's polyp count and size ({fact key: value}), or "" when it predicts
    no polyp."""
    try:
        n = int(str(facts.get("polyp_count") or "0").strip() or 0)
    except ValueError:
        return ""
    if n <= 0:
        return ""
    size = _SIZES.get(str(facts.get("polyp_size") or "").strip().lower())
    return f"{_COUNT_WORDS.get(n, n)} polyp{'' if n == 1 else 's'} identified" + (f", {size}" if size else "")


# Instruments, the same way. On 178 held-out images with a labelled instrument count the fact classifier was right on 177 (it found
# all 51 images that have an instrument); the generated "instruments visible?" answer found 46 of the 51.
def instrument_detail(facts):
    """"One surgical instrument present" from the fact classifier's instrument count or type ({fact key: value}), or "" when it
    predicts none."""
    try:
        n = int(str(facts.get("instrument_count") or "0").strip() or 0)
    except ValueError:
        n = 0
    if n <= 0:
        n = 1 if str(facts.get("instrument_presence") or "").strip() else 0
    if n <= 0:
        return ""
    return f"{_COUNT_WORDS.get(n, n)} surgical instrument{'' if n == 1 else 's'} present"
