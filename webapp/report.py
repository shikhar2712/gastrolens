"""The downloadable report of the web demo: one A4 page for one image, laid out like the endoscopy procedure reports that
hospitals and endoscopy-reporting software produce (the layout of sample colonoscopy and gastroscopy reports was studied for this):
a title with the organisation mark and a heavy rule, a two-column patient and examination block, bold run-in section headings
(Introduction, Procedure, Findings, Summary, Recommendations), the photographs stacked in a right-hand column in numbered boxes with
captions, and a signature line and a "Page 1 of 1" footer.

Only what the model produced is in it: the image beside the Grad-CAM of one finding, the model's standard reading of the image
(procedure, abnormalities, polyps, instruments, landmarks), and the automatic image checks that returned a value. Fields a real
report has but this tool cannot know (indications, sedation, bowel preparation, endoscopist, procedure codes) are left out, not
invented; the clinician's review is a blank line to sign. The footer says on every page that it is AI generated, a research
prototype and not for clinical decisions. If patient details were given (all optional) they head the report; if not, there is no
patient field at all.

It is always one page: if the content does not fit, the text and the pictures shrink, and as a last resort the findings are cut
and say how many were left out.

The server builds it from a fresh model run on the image, never from text the browser sends, so what is in the file is what the
model said. Patient details come in with the request, go straight into the PDF and are neither logged nor stored; they are kept
out of the file name and the PDF's metadata.
"""
import base64
import io
import os
import re
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.shapes import Circle, Drawing, Line
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.doctemplate import LayoutError

from webapp import questions

SEX_CHOICES = ("Female", "Male", "Other")
MAX_NAME = 80
NOTICE = "AI generated. Research prototype, not a medical device. Not for diagnosis or clinical decisions."
FIGURE_BOX = (60 * mm, 44 * mm)  # the largest a photograph may be at full size; the proportions of the picture are kept
# (text and picture scale, findings shown), tried in order: the first that fits one page wins, so a short report is set larger and
# fills the page, and a long one is set smaller, and as a last resort its findings are cut
ATTEMPTS = ((1.12, None), (1.04, None), (1.0, None), (0.93, None), (0.86, None), (0.78, None), (0.78, 4), (0.7, 3))

INK, MUTED, RULE = colors.HexColor("#151a33"), colors.HexColor("#566088"), colors.HexColor("#c9cfe3")
ACCENT = colors.HexColor("#3b54d6")

_font = {}


def _font_dirs():
    dirs = ["/usr/share/fonts/truetype/dejavu"]
    try:  # matplotlib ships DejaVu Sans
        import matplotlib
        dirs.insert(0, os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except ImportError:
        pass
    return dirs


def fonts():
    """(regular, bold, unicode): DejaVu Sans when it can be found (Latin, Greek, Cyrillic), else the built-in Helvetica, which
    only has Latin-1, in which case characters outside it are replaced with '?' rather than drawn as empty boxes."""
    if not _font:
        reg, bold, uni = "Helvetica", "Helvetica-Bold", False
        for d in _font_dirs():
            r, b = os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")
            if os.path.exists(r) and os.path.exists(b):
                pdfmetrics.registerFont(TTFont("DejaVuSans", r))
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", b))
                pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold", italic="DejaVuSans", boldItalic="DejaVuSans-Bold")
                reg, bold, uni = "DejaVuSans", "DejaVuSans-Bold", True
                break
        _font.update(reg=reg, bold=bold, uni=uni)
    return _font["reg"], _font["bold"], _font["uni"]


def clean_text(s, limit):
    """Patient-supplied or model-supplied text made safe to print: control characters out, runs of space collapsed, length capped."""
    s = re.sub(r"[\x00-\x1f\x7f-\x9f  ]", " ", str(s or ""))
    return " ".join(s.split())[:limit]


def clean_patient(name, age, sex):
    """Validated patient details -> {'Name':..., 'Age':..., 'Sex':...} with only the filled fields. Raises ValueError with a message
    written for the person filling the form."""
    out = {}
    name = clean_text(name, MAX_NAME + 1)
    if len(name) > MAX_NAME:
        raise ValueError(f"Patient name is too long (limit {MAX_NAME} characters).")
    if name:
        out["Name"] = name
    age = (age or "").strip()
    if age:
        if not re.fullmatch(r"\d{1,3}", age) or int(age) > 130:
            raise ValueError("Age must be a whole number from 0 to 130.")
        out["Age"] = f"{int(age)} years"
    sex = (sex or "").strip()
    if sex:
        if sex not in SEX_CHOICES:
            raise ValueError("Sex must be one of: " + ", ".join(SEX_CHOICES) + ".")
        out["Sex"] = sex
    return out


def _text(s):
    """Text for a Paragraph: escaped, and reduced to Latin-1 when only Helvetica is available."""
    s = str(s or "")
    if not fonts()[2]:
        s = s.encode("latin-1", "replace").decode("latin-1")
    return escape(s)


def _data_url_bytes(url):
    if not url or "," not in url:
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except ValueError:
        return None


def _fit(raw, box):
    """An Image flowable of these bytes scaled to fit the box, keeping its proportions."""
    with PILImage.open(io.BytesIO(raw)) as im:
        w, h = im.size
    k = min(box[0] / w, box[1] / h)
    return Image(io.BytesIO(raw), width=w * k, height=h * k)


def _original_jpeg(path, max_side=1000):
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            k = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _styles(k):
    """Paragraph styles, every size and gap multiplied by k (the fill-the-page scale)."""
    reg, bold, _ = fonts()

    def ps(name, size, lead, **kw):
        return ParagraphStyle(name, fontName=kw.pop("font", reg), textColor=kw.pop("color", INK), fontSize=size * k, leading=lead * k, **kw)

    return {
        "title": ps("title", 15.5, 19, font=bold),
        "tsub": ps("tsub", 9.5, 13, color=MUTED, spaceBefore=1 * k),
        "brand": ps("brand", 17, 20, font=bold, color=ACCENT, alignment=TA_RIGHT),
        "ik": ps("ik", 9.2, 12.5, font=bold),
        "iv": ps("iv", 9.2, 12.5),
        "body": ps("body", 9.6, 13.2, spaceAfter=4 * k),
        "head": ps("head", 10.2, 13.5, font=bold, spaceBefore=6 * k, spaceAfter=2 * k),
        "bul": ps("bul", 9.6, 13, leftIndent=11, bulletIndent=1, spaceAfter=1.5 * k),
        "note": ps("note", 8, 10.8, color=MUTED, spaceAfter=3 * k),
        "cap": ps("cap", 8, 10.5, color=MUTED),
        "num": ps("num", 8.5, 10, font=bold, alignment=1),
        "gh": ps("gh", 9, 12, font=bold),
        "k": ps("k", 8.6, 11.3, color=MUTED),
        "v": ps("v", 9.2, 11.3, font=bold, alignment=2),
        "small": ps("small", 8.6, 12, color=MUTED),
        "sig": ps("sig", 7.8, 10, color=MUTED),
    }


REGIONS = {"upper-left", "upper-center", "upper-right", "center-left", "center", "center-right", "lower-left", "lower-center", "lower-right"}


def display_value(value):
    """A fact's value as printed: a list of regions reads with commas, and all nine of them as 'all 9 regions'."""
    parts = [p.strip() for p in str(value).split(";") if p.strip()]
    return "all 9 regions" if set(parts) == REGIONS else ", ".join(parts)


def _sentence(text):
    """The model's answer as a line of the report: first letter capital, nothing else changed."""
    t = " ".join(str(text or "").split())
    return t[:1].upper() + t[1:]


def _logo(size):
    """The mark: a scope's view, as a ring with an inner ring and a centre dot."""
    d = Drawing(size, size)
    c = size / 2
    d.add(Circle(c, c, c - 2, strokeColor=ACCENT, strokeWidth=2.2, fillColor=None))
    d.add(Circle(c, c, c * 0.46, strokeColor=ACCENT, strokeWidth=1.6, fillColor=None))
    d.add(Circle(c, c, 1.8, fillColor=ACCENT, strokeColor=None))
    for x1, y1, x2, y2 in ((c, 0, c, 5), (c, size - 5, c, size), (0, c, 5, c), (size - 5, c, size, c)):
        d.add(Line(x1, y1, x2, y2, strokeColor=ACCENT, strokeWidth=2))
    return d


def _header(st, width, k):
    """The title on the left, the organisation mark on the right, a heavy rule under both."""
    mark = Table([[_logo(30 * k), Paragraph("GastroLens", st["brand"])]], colWidths=[36 * k, width * 0.34 - 36 * k], hAlign="RIGHT")
    mark.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    t = Table([[[Paragraph("Endoscopy Image Analysis Report", st["title"]), Paragraph("AI generated. Automatic analysis of one image.", st["tsub"])], mark]],
              colWidths=[width * 0.66, width * 0.34])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 7 * k), ("LINEBELOW", (0, 0), (-1, -1), 2, INK)]))
    return t


def _info(left, right, st, width, k):
    """The two-column block under the title: patient on the left, examination on the right, a rule under it (as in the sample reports)."""
    n = max(len(left), len(right))
    pad = lambda rows: rows + [("", "")] * (n - len(rows))
    left, right = pad(left), pad(right)

    def lab(x):
        return Paragraph(_text(x + ":") if x else "", st["ik"])

    rows = [[lab(a), Paragraph(_text(b), st["iv"]), lab(c), Paragraph(_text(d), st["iv"])] for (a, b), (c, d) in zip(left, right)]
    t = Table(rows, colWidths=[width * 0.13, width * 0.37, width * 0.17, width * 0.33])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5 * k), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * k),
                           ("LINEBELOW", (0, -1), (-1, -1), 2, INK), ("BOTTOMPADDING", (0, -1), (-1, -1), 6 * k)]))
    return t


def _photo(number, raw, caption, st, colw, k):
    """One photograph in a thin frame with a numbered box and a caption under it, as the sample reports set them."""
    f = min(k, 1.0)
    img = _fit(raw, (FIGURE_BOX[0] * f, FIGURE_BOX[1] * f))
    frame = Table([[img]], hAlign="LEFT")
    frame.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    box = Table([[Paragraph(str(number), st["num"])]], colWidths=[11], rowHeights=[11])  # a small square, whatever the caption's length
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    cap = Table([[box, Paragraph(_text(caption), st["cap"])]], colWidths=[14, colw - 14], hAlign="LEFT")
    cap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return [frame, Spacer(1, 3), cap, Spacer(1, 9 * k)]


def _bullets(lines, st):
    return [Paragraph(_text(line), st["bul"], bulletText="\u2022") for line in lines]


def _group_table(title, rows, st, colw, k):
    """One group of checks: a bold heading with a rule, then label on the left and value on the right, one row per check."""
    head = Table([[Paragraph(_text(title), st["gh"])]], colWidths=[colw])
    head.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, INK), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    body = Table([[Paragraph(_text(lbl), st["k"]), Paragraph(_text(val), st["v"])] for lbl, val in rows], colWidths=[colw * 0.52, colw * 0.48])
    body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                              ("TOPPADDING", (0, 0), (-1, -1), 2.6 * k), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2 * k),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return [head, body, Spacer(1, 7 * k)]


def _checks(facts, groups, st, width, k):
    """The automatic checks that returned a value, in their groups, in two columns."""
    by_key = {f["key"]: f for f in facts if f.get("value")}
    listed = {key for _, keys in groups for key in keys}
    blocks = [(title, [(by_key[key]["label"], display_value(by_key[key]["value"])) for key in keys if key in by_key]) for title, keys in groups]
    blocks.append(("Other", [(f["label"], display_value(f["value"])) for f in facts if f.get("value") and f["key"] not in listed]))
    blocks = [b for b in blocks if b[1]]
    if not blocks:
        return Paragraph("No check returned a value for this image.", st["small"])
    gap = 22
    colw = (width - gap) / 2
    cols, heights = [[], []], [0, 0]
    for title, rows in blocks:  # each group goes under the shorter column, so the two come out about level
        c = 0 if heights[0] <= heights[1] else 1
        cols[c] += _group_table(title, rows, st, colw, k)
        heights[c] += len(rows) + 2
    outer = Table([[cols[0], "", cols[1]]], colWidths=[colw, gap, colw])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return outer


def _signature(st, width, k, when):
    """A blank line to sign, as on the sample reports: this one is for the reviewing clinician; the model does not sign."""
    w = [width * 0.38, width * 0.04, width * 0.24, width * 0.04, width * 0.30]
    t = Table([["", "", "", "", ""], [Paragraph("Reviewed by (clinician)", st["sig"]), "", Paragraph("Date", st["sig"]), "", Paragraph("Signature", st["sig"])]],
              colWidths=w, rowHeights=[20 * k, None])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (0, 0), 0.7, INK), ("LINEBELOW", (2, 0), (2, 0), 0.7, INK), ("LINEBELOW", (4, 0), (4, 0), 0.7, INK),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 1), (-1, 1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    stamp = Paragraph(_text(f"Report generated automatically by GastroLens on {when.strftime('%d %b %Y at %H:%M %Z').strip()}. "
                            "It has not been signed or reviewed by a clinician."), st["sig"])
    return [t, Spacer(1, 4), stamp]


def _build(image_path, image_label, analysis, patient, when, k, limit):
    reg, bold, _ = fonts()
    st = _styles(k)
    buf = io.BytesIO()
    margin = 17 * mm
    doc = BaseDocTemplate(buf, pagesize=A4, title="GastroLens image analysis report (AI generated)", author="GastroLens",
                          creator="GastroLens", subject="AI-generated analysis of one endoscopy image")
    width = A4[0] - 2 * margin

    def page(canvas, d):
        canvas.saveState()
        canvas.setStrokeColor(INK)
        canvas.setLineWidth(0.8)
        canvas.line(margin, 15 * mm, A4[0] - margin, 15 * mm)
        canvas.setFont(reg, 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 10.5 * mm, NOTICE)
        canvas.drawRightString(A4[0] - margin, 10.5 * mm, "Page 1 of 1")
        canvas.restoreState()

    facts = analysis["facts"]
    raw = list(analysis["findings"])
    conf = analysis.get("confidence") or {}
    for i, key in ((1, "abnormality"), (2, "polyp")):  # the abnormality and polyp findings carry the model's own probability
        if len(raw) > i and raw[i] and conf.get(key) is not None:
            raw[i] = f"{raw[i]} (model confidence {'99%+' if conf[key] >= 99 else str(conf[key]) + '%'})"
    findings = [_sentence(a) for a in dict.fromkeys(a for a in raw if a)]
    proc = next((f["value"] for f in facts if f["key"] == "procedure_type" and f["value"]), "")

    left_info = list(patient.items()) + [("Image", image_label)]
    right_info = [("Procedure", (proc.capitalize() + " (model prediction)") if proc else "Not predicted"),
                  ("Date", when.strftime("%d %b %Y, %H:%M %Z").strip()), ("Prepared by", "GastroLens (automatic analysis)"),
                  ("Status", "Awaiting clinician review")]

    # body, left column: the text sections, with bold run-in headings
    colw_l = width * 0.60
    gap = 16
    colw_r = width - colw_l - gap
    shown = findings[:limit] if limit and len(findings) > limit else findings
    extra = len(findings) - len(shown)
    left = [Paragraph("<b>Introduction:</b> " + _text("Automatic analysis of one endoscopy image by GastroLens, a research model. "
                                                    "The picture is read by the model alone; no clinical history was available to it."), st["body"])]
    if proc:
        left.append(Paragraph("<b>Procedure:</b> " + _text(f"{proc.capitalize()}, as predicted by the model from the image."), st["body"]))
    left.append(Paragraph("Findings:", st["head"]))
    left += _bullets(shown, st) if shown else [Paragraph("No findings.", st["small"])]
    if extra:
        left.append(Paragraph(f"and {extra} more finding{'s' if extra != 1 else ''} not shown", st["small"]))
    positives = [f for f in findings[1:] if questions.is_positive(f)]  # the first finding is the procedure, which is not a problem
    left.append(Paragraph("Summary:", st["head"]))
    left += _bullets(positives, st) if positives else [Paragraph("The model reported no abnormality, polyp, instrument or landmark.", st["small"])]
    left.append(Paragraph("Recommendations:", st["head"]))
    left += _bullets(["Review of the image and of this report by a qualified clinician. The model makes no clinical recommendation."], st)

    # right column: the photographs
    right = []
    original = _original_jpeg(image_path)
    right += _photo(1, original, "Original image", st, colw_r, k)
    cam_raw = _data_url_bytes(analysis["cam"]["image"]) if analysis.get("cam") else None
    if cam_raw:
        right += _photo(2, cam_raw, f"Grad-CAM overlay for the finding: {_sentence(analysis.get('cam_answer'))}. Warmer colour means more influence.", st, colw_r, k)
    else:
        right.append(Paragraph("Grad-CAM was not available for this image.", st["small"]))

    body = Table([[left, "", right]], colWidths=[colw_l, gap, colw_r])
    body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))

    n_filled = sum(1 for f in facts if f.get("value"))
    story = [_header(st, width, k), Spacer(1, 7 * k), _info(left_info, right_info, st, width, k), Spacer(1, 9 * k), body,
             Paragraph("Image checks:", st["head"]),
             Paragraph(_text(f"{n_filled} of {len(facts)} automatic checks returned a value. They are the model's own predictions: they can be "
                             "wrong and can disagree with each other and with the findings."), st["note"]),
             _checks(facts, analysis["fact_groups"], st, width, k), Spacer(1, 10 * k)] + _signature(st, width, k, when)

    frame = Frame(margin, 19 * mm, width, A4[1] - 15 * mm - 19 * mm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=page)])
    doc.build(story)
    return buf.getvalue(), doc.page


def build_pdf(*, image_path, image_label, analysis, patient, when):
    """The report as PDF bytes, always one page, set as large as will fit so that it uses the page.

    analysis: {'facts', 'fact_groups', 'findings': [answer, ...], 'cam', 'cam_answer'}: the model's reading of the image, with the
    Grad-CAM of one finding (cam may be None). patient: output of clean_patient (possibly empty). when: an aware datetime."""
    pdf = b""
    for k, limit in ATTEMPTS:
        try:
            pdf, pages = _build(image_path, image_label, analysis, patient, when, k, limit)
        except LayoutError:  # something did not fit in the frame at this size: try a smaller one
            continue
        if pages == 1:
            return pdf
    if not pdf:
        raise RuntimeError("the report does not fit on a page at any size")
    return pdf
