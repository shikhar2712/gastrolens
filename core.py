"""The model side of the Space: load the released weights, read one image, build the report. No web framework in here.

The analysis mirrors webapp/server.py (`_analysis`): the five standard questions in one batch, the 17 facts, the Grad-CAM of the abnormality
decision, the classifier's polyp and instrument findings and the confidence of each. The consistency step is left out on purpose: it only
corrects location and polyp-type answers, none of the five standard questions is one, and it would need the 12 MB training-question table.
"""
import hashlib
import io
import os
import tempfile
from datetime import datetime

import torch
from PIL import Image, ImageOps

os.environ.setdefault("GASTROLENS_LEAN", "1")  # memory-lean start-up (kvqa/model.py); must be set before the model code is imported

from kvqa.predict import Predictor
from webapp import confidence, questions
from webapp.gradcam import explain

DEFAULT_WEIGHTS_REPO = "shikharr27/gastrolens-weights"  # not secret; the repo is private, so downloading it needs HF_TOKEN
RELEASE_SHA256 = "e9401ab00739257f35918514d09ad37559da695b8f16cd2ec6b1c343299be345"  # experiments/kvasir_vqa_x1_final/checkpoints/best.pt
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
NOTICE = "Research prototype, not a medical device: do not use for clinical decisions. Trained on a CC BY-NC 4.0 dataset: non-commercial use only."

ATTR_LABELS = {
    "abnormality_color": "Abnormality colour", "abnormality_location": "Abnormality location", "abnormality_presence": "Abnormality present",
    "box_artifact_presence": "Box artifact present", "finding_count": "Number of findings", "finding_presence": "Finding present",
    "instrument_count": "Number of instruments", "instrument_location": "Instrument location", "instrument_presence": "Instrument present",
    "landmark_color": "Landmark colour", "landmark_location": "Landmark location", "landmark_presence": "Landmark present",
    "polyp_count": "Number of polyps", "polyp_removal_status": "Polyp removal status", "polyp_size": "Polyp size", "polyp_type": "Polyp type",
    "procedure_type": "Procedure type", "text_presence": "Text present",
}


class ImageError(ValueError):
    """The upload cannot be used; the message is written for the person who uploaded it."""


def weights_dir():
    """The folder holding checkpoints/best.pt and model/*.json: a local path (GASTROLENS_WEIGHTS_DIR) or a Hugging Face model repo
    (GASTROLENS_WEIGHTS_REPO), downloaded once."""
    local = os.environ.get("GASTROLENS_WEIGHTS_DIR")
    if local:
        return local
    repo = os.environ.get("GASTROLENS_WEIGHTS_REPO") or DEFAULT_WEIGHTS_REPO
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:  # the cause is usually a missing or wrong token on a private repo; say so without printing anything secret
        hint = "no HF_TOKEN is set" if not os.environ.get("HF_TOKEN") else "the HF_TOKEN may be wrong or may not have read access to it"
        raise RuntimeError(f"Could not download the weights from the Hugging Face repo {repo!r}: {hint} ({type(e).__name__}).") from None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictor(device="cpu", threads=None):
    """The released model, checked against its published SHA-256, on this device."""
    exp = weights_dir()
    ckpt = os.path.join(exp, "checkpoints", "best.pt")
    if _sha256(ckpt) != RELEASE_SHA256:
        raise RuntimeError("The downloaded checkpoint is not the released one (SHA-256 mismatch); refusing to run it.")
    if threads:
        torch.set_num_threads(threads)
    return Predictor(exp, device=device, route=None)


def read_image(raw):
    """Validate an upload and return an upright RGB copy written to a temp file (the model reads from a path). Raises ImageError."""
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageError(f"That file is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ImageError("That image has too many pixels.")
            im = ImageOps.exif_transpose(im).convert("RGB")  # a phone photo with a rotation tag is shown upright, so it is read upright
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ImageError("Could not read that file as an image.")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    im.save(path, "PNG")
    return path, im


def analyse(pr, image_path):
    """Everything the page shows for one image."""
    outs = pr.answer([(image_path, q) for q in questions.STANDARD])
    answers = [o["answer"] for o in outs]
    cam_answer = answers[questions.STANDARD.index(questions.CAM_QUESTION)]
    facts = [{"key": k, "label": ATTR_LABELS.get(k, k), "value": v} for k, v in outs[0]["fact_classifiers"].items()]
    by_key = {f["key"]: f["value"] for f in facts}
    abnormality = (by_key.get("abnormality_presence") or "").split(";")[0].strip()
    cam = None
    try:  # the heat is for the classifier's own abnormality decision when it names one, else for the generated answer
        cam = explain(pr, image_path, questions.CAM_QUESTION, cam_answer, ("abnormality_presence", abnormality) if abnormality else None)
    except Exception:
        cam = None  # the answer stands without a heatmap
    polyp = questions.polyp_detail(by_key)
    if polyp:
        answers[questions.STANDARD.index(questions.POLYP_QUESTION)] = polyp[:1].lower() + polyp[1:]
    instrument = questions.instrument_detail(by_key)
    if instrument:
        answers[questions.STANDARD.index(questions.INSTRUMENT_QUESTION)] = instrument[:1].lower() + instrument[1:]
    return {"facts": facts, "fact_groups": questions.FACT_GROUPS, "findings": answers, "detail": polyp, "cam": cam, "cam_answer": cam_answer,
            "confidence": confidence.summarize(confidence.head_probs(pr, image_path), by_key)}


def build_report(image_path, analysis):
    """The one-page PDF for this analysis (no patient details: the public demo does not ask for any)."""
    from webapp import report
    return report.build_pdf(image_path=image_path, image_label="Uploaded image", analysis=analysis, patient={}, when=datetime.now().astimezone())
