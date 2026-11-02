"""Grad-CAM for the demo: where in the image the model looked to produce the answer it gave.

The model reads the image as a 24 x 24 grid of patch tokens (SigLIP2, 384 px, 16 px patches) that the attribute branch and the
question-conditioned fusion both attend to. The score is the sum of the decoder's pre-softmax logits for the tokens of the answer
that is shown (teacher-forced), so it also explains an answer the consistency step changed. Each patch is weighted by
gradient x activation, summed over channels, with a ReLU to keep what raised the score: the per-patch form of Grad-CAM (HiResCAM,
Draelos and Carin 2020). The classic form, which averages the gradient over the grid per channel, and a log-probability score
were both tried on the example images and came out as diffuse speckle on this model, so they are not used.

One pass gives a 24 x 24 map, which is blocky when stretched over the picture. So the map is computed on 9 copies of the image
shifted by up to 8 px (one batch), each map is shifted back into place, and they are averaged: that cancels the patch-grid
squares and one-off spikes, and the shifted tokenisations add real detail finer than one patch. The result is smoothed
widely (so it reads as regions, not pin-points), the black frame around the endoscope view is cleared (the model often reads the
burned-in timestamp there, and tissue is never in it), and the map is scaled to its own maximum and flattened with a power below
1, so that weaker regions beside the strongest still show: a scattered map where the model's attention is scattered. 64 x 64 grid.

The map is relative and indicative: the model was trained on answers and fact labels, never on where things are, so it is not a
segmentation and says nothing about correctness.

Everything runs under torch.autograd.grad, so no parameter .grad is ever filled in and nothing accumulates between requests.
"""
import base64
import io
import math

import numpy as np
import torch
from PIL import Image

from kvqa.data import BOS, EOS

# Opacity of the heat by relative influence. Below 0.3 it is nil: where the model hardly looked the picture is left exactly as it is
# (the same grey the page draws underneath), so nothing about the picture changes as the heat is revealed; colour begins where the
# influence does.
ALPHA = ([0.0, 0.30, 0.50, 1.0], [0.0, 0.0, 0.55, 0.88])
MAX_SIDE = 768  # longest side of the returned picture
SHIFT = 8  # px, in the model's 384 px input; the 9 copies are shifted by -SHIFT, 0, +SHIFT in x and y
OUT_GRID = 64  # size of the map that is returned
SMOOTH_PX = 20.0  # Gaussian sigma of the final smoothing, in the model's 384 px input (compared with 12 and 26 on the examples)
GAMMA = 0.7  # the scaled map is raised to this: below 1 lifts the weaker regions next to the strongest
DARK_LUMA, DARK_WIN = 12.0, 25  # a window of DARK_WIN px that is mostly darker than this (0-255) is the black frame, not tissue
EDGE_SOFT = 41  # the clear rim fades out over about this many px, so the heat tapers to nothing instead of ending in an edge
EDGE_PX = 26  # the rim of the tissue beside the frame is cleared too: its patches mix frame and tissue and light up on their own


def _blur(c, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32, device=c.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = torch.nn.functional.pad(c[None, None], (r, r, r, r), mode="reflect")
    c = torch.nn.functional.conv2d(c, k.view(1, 1, 1, -1))
    return torch.nn.functional.conv2d(c, k.view(1, 1, -1, 1))[0, 0]


def _tissue(x):
    """(S, S) weights in [0, 1]: 0 in the black frame around the endoscope view, on text drawn on it, and on a thin rim of tissue
    beside it. x: the model's input, (3, S, S) in [-1, 1]. A window counts as frame when most of its pixels are almost black, so a
    few bright text pixels on a black bar are judged by what is around them. Only dark regions that reach the edge of the picture are
    frame: a dark lumen in the middle of the tissue is not, and the rim is cleared only around the frame."""
    luma = ((x * 0.5 + 0.5) * 255 * torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)[:, None, None]).sum(0)
    dark = (luma < DARK_LUMA).float()[None, None]
    frac = torch.nn.functional.avg_pool2d(dark, DARK_WIN, stride=1, padding=DARK_WIN // 2, count_include_pad=False)
    mostly_dark = (frac > 0.5).float()
    seed = torch.zeros_like(mostly_dark)
    seed[..., 0, :] = seed[..., -1, :] = seed[..., :, 0] = seed[..., :, -1] = 1.0
    frame = seed * mostly_dark
    for _ in range(40):  # grow from the picture's edge through the dark region: what it reaches is the frame
        grown = torch.nn.functional.max_pool2d(frame, 15, stride=1, padding=7) * mostly_dark
        if torch.equal(grown, frame):
            break
        frame = grown
    near = torch.nn.functional.max_pool2d(frame, 2 * EDGE_PX + 1, stride=1, padding=EDGE_PX)  # the frame and the rim beside it
    soft = torch.nn.functional.avg_pool2d(near, EDGE_SOFT, stride=1, padding=EDGE_SOFT // 2, count_include_pad=False)  # no hard edge in the map
    return (1 - soft)[0, 0]


def cam_grid(pr, image_path, question, answer, attr=None):
    """Relative attention over the image, shape (OUT_GRID, OUT_GRID), values in [0, 1]; None if nothing raised the score.
    attr=(fact class, label), e.g. ("abnormality_presence", "polyp"): the score is that classifier's own logit for the label,
    instead of the decoder's logits for the answer's tokens."""
    m, dev = pr.model, pr.device
    img = pr.load_image(image_path).unsqueeze(0).float().div(255).sub(0.5).div(0.5).to(dev)  # (1, 3, S, S)
    S = img.size(-1)
    shifts = [(sy, sx) for sy in (-SHIFT, 0, SHIFT) for sx in (-SHIFT, 0, SHIFT)]
    padded = torch.nn.functional.pad(img, (SHIFT,) * 4, mode="replicate")
    batch = torch.cat([padded[:, :, SHIFT + sy:SHIFT + sy + S, SHIFT + sx:SHIFT + sx + S] for sy, sx in shifts])  # (9, 3, S, S)
    B = batch.size(0)
    q = pr.qtok([question], padding=True, truncation=True, max_length=pr.cfg["data"]["max_question_len"], return_tensors="pt")
    ids = [BOS] + pr.atok.encode(answer).ids[:pr.cfg["data"]["max_answer_len"] - 2] + [EOS]
    ans = torch.tensor([ids], device=dev).expand(B, -1)
    with torch.no_grad():
        v = m.v_proj(m.vision(pixel_values=batch).last_hidden_state) + m.pos2d
    v = v.detach().requires_grad_(True)  # the tokens both branches attend to; the vision tower itself is not differentiated
    with torch.enable_grad():
        a = m.attr_queries.expand(B, -1, -1)
        for blk in m.attr_blocks:
            a = blk(a, v)
        a = m.attr_norm(a)
        if attr:
            cls, label = attr
            score = m.attr_heads[cls](a[:, m.attr_classes.index(cls)])[:, m.attr_schema[cls]["labels"].index(label)].float().sum()
        else:
            mem, pad = m.fuse(v, a, torch.arange(B, device=dev), q.input_ids.to(dev).expand(B, -1), q.attention_mask.to(dev).expand(B, -1))
            logits = m.decode(ans[:, :-1], mem, pad)
            score = logits.float().gather(-1, ans[:, 1:, None]).sum()  # one independent score per copy; their sum has per-copy gradients
        (g,) = torch.autograd.grad(score, v)
    cam = torch.relu((v.detach() * g).sum(-1))  # (B, n*n)
    n = int(round(math.sqrt(cam.size(1))))
    up = torch.nn.functional.interpolate(cam.reshape(B, 1, n, n).float(), size=(S, S), mode="bilinear", align_corners=False)[:, 0]
    acc, cnt = torch.zeros(S, S, device=dev), torch.zeros(S, S, device=dev)
    for k, (sy, sx) in enumerate(shifts):  # copy k shows the picture moved by (sy, sx): move its map back
        ys, ye, xs, xe = max(0, -sy), min(S, S - sy), max(0, -sx), min(S, S - sx)
        acc[ys + sy:ye + sy, xs + sx:xe + sx] += up[k, ys:ye, xs:xe]
        cnt[ys + sy:ye + sy, xs + sx:xe + sx] += 1
    cam = _blur(acc / cnt.clamp(min=1), SMOOTH_PX) * _tissue(img[0])
    cam = torch.nn.functional.adaptive_avg_pool2d(cam[None, None], OUT_GRID)[0, 0].cpu().numpy()
    top = float(cam.max())
    return None if not np.isfinite(top) or top <= 0 else (cam / top) ** GAMMA


def jet(h):
    """The classic jet colormap (dark blue, cyan, green, yellow, red) for values in [0, 1]; returns (..., 3) in 0-255."""
    h = np.asarray(h, dtype=np.float32)
    return np.stack([np.clip(1.5 - np.abs(4 * h - 3), 0, 1), np.clip(1.5 - np.abs(4 * h - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * h - 1), 0, 1)], -1) * 255


def render(image_path, grid, max_side=MAX_SIDE):
    """The image in black and white with the heat drawn over it, as a JPEG data URL, at the original aspect ratio (the model
    squashes every image to a square, so the grid is stretched back over the full picture, not letterboxed). The grey uses the
    same luminance weights as CSS grayscale(1), so the page can draw the plain grey picture under this one and they line up."""
    im = Image.open(image_path).convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    heat = np.asarray(Image.fromarray(grid.astype(np.float32), mode="F").resize(im.size, Image.BICUBIC))
    heat = np.clip(heat, 0.0, 1.0)
    rgb = np.asarray(im).astype(np.float32)
    grey = (rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(-1, keepdims=True)
    alpha = np.interp(heat, *ALPHA)[..., None]
    out = np.repeat(grey, 3, -1) * (1 - alpha) + jet(heat) * alpha
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_overlay(image_path, grid, max_side=640):
    """Only the heat, as a transparent PNG data URL with the picture's proportions: colour where the model's influence is meaningful
    (the same opacity ramp as render), fully transparent elsewhere. The page lays it over the picture itself, so what is behind the
    heat is the original image, not a redraw of it."""
    with Image.open(image_path) as im:
        w, h = im.size
    s = min(1.0, max_side / max(w, h))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    heat = np.clip(np.asarray(Image.fromarray(grid.astype(np.float32), mode="F").resize(size, Image.BICUBIC)), 0.0, 1.0)
    alpha = np.interp(heat, *ALPHA)
    rgba = np.dstack([jet(heat).clip(0, 255), alpha * 255]).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def explain(pr, image_path, question, answer, attr=None):
    """{'image': grey picture with the heat (JPEG data URL, for the report), 'overlay': the heat alone (transparent PNG data URL, for
    the page), 'grid': n x n list, 'peak': [x, y] as fractions of the width and height}, or None."""
    if not hasattr(pr, "model"):  # a predictor without a model (the test stub)
        return None
    grid = cam_grid(pr, image_path, question, answer, attr)
    if grid is None:
        return None
    n = grid.shape[0]
    iy, ix = np.unravel_index(int(grid.argmax()), grid.shape)
    return {"image": render(image_path, grid), "overlay": render_overlay(image_path, grid), "grid": np.round(grid, 3).tolist(),
            "peak": [round((ix + 0.5) / n, 3), round((iy + 0.5) / n, 3)]}
