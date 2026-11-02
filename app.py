import base64
import io
import os
import threading
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

import core
from webapp.report import display_value

st.set_page_config(page_title="GastroLens", layout="wide")

for _name in ("HF_TOKEN", "GASTROLENS_WEIGHTS_REPO"):  # Streamlit Cloud keeps these under Secrets; the model code reads environment variables
    try:
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = str(st.secrets[_name])
    except Exception:
        pass  # no secrets configured: running locally with plain environment variables

st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 2rem; }
.answer { border-radius: 14px; padding: 18px 22px; background: rgba(79,102,234,.14); margin-top: .5rem; }
.answer .tag { display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; border: 1px solid #8fa2ff; margin-bottom: 8px; }
.answer .head { font-family: Georgia, "Times New Roman", serif; font-size: 30px; font-weight: 600; line-height: 1.2; }
.answer .detail { font-size: 17px; font-weight: 600; margin-top: 6px; }
.fine { font-size: 13px; opacity: .75; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("GastroLens")
st.write("Upload a GI endoscopy image. See what the model finds, and exactly where it looked.")
st.warning("For research use only. Not a medical device: not for diagnosis or clinical decisions. "
           "Do not upload images of real patients: uploads are processed on this server.")


class Warmup:
    """Loads the model on a background thread, so it is usually ready by the time a visitor has picked an image.
    The thread never touches Streamlit; `get()` waits for it and raises the load error, and a failed load is retried on the next call."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        self.model, self.error = None, None
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.model = core.load_predictor(device="cpu", threads=int(os.environ.get("GASTROLENS_THREADS", "2")))
        except Exception as e:  # reported to the visitor by get()
            self.error = e
        finally:
            self.done.set()

    @property
    def ready(self):
        return self.done.is_set() and self.error is None

    def get(self):
        self.done.wait()
        with self._lock:
            if self.error is not None:
                err = self.error
                self._start()  # let the next click try again (a fixed token, a network blip)
                raise err
            return self.model


@st.cache_resource
def warmup():
    return Warmup()


def predictor():
    return warmup().get()


warmup()  # start loading as soon as the first visitor opens the page


@st.cache_resource
def analysis_lock():
    """One analysis at a time: the host has little memory, and two Grad-CAM runs at once would double the working set."""
    return threading.Lock()


def data_url_image(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")


def with_heat(original, overlay_url, black_and_white):
    base = original.convert("L").convert("RGB") if black_and_white else original
    base = base.convert("RGBA")
    return Image.alpha_composite(base, data_url_image(overlay_url).resize(base.size)).convert("RGB")


def fit(widget, *args, **kw):
    """Call a Streamlit widget so that it fills its column on both old and new Streamlit versions."""
    try:
        return widget(*args, width="stretch", **kw)
    except TypeError:
        return widget(*args, use_container_width=True, **kw)


def pct(v):
    return "99%+" if v >= 99 else f"{v}%"


SAMPLES = sorted(Path(__file__).parent.glob("samples/sample*.jpg"))


@st.cache_data
def thumbnail(path, side=150):
    """A centred square crop, so every sample tile is the same size and the buttons line up."""
    with Image.open(path) as im:
        return ImageOps.fit(im.convert("RGB"), (side, side))


def fresh_start():
    """Forget the result and the chosen sample, and empty the uploader (a new widget key gives it a clean state)."""
    st.session_state.update(result=None, sample=None, up=st.session_state.get("up", 0) + 1)


upload = st.file_uploader("Choose an endoscopy image (JPG or PNG)", type=["jpg", "jpeg", "png"], key=f"upload{st.session_state.get('up', 0)}")
sample = st.session_state.get("sample")
if upload is None and sample is None:
    if SAMPLES:
        st.markdown("**Or try a sample image**")
        cols = st.columns(len(SAMPLES))
        for i, (col, path) in enumerate(zip(cols, SAMPLES), 1):
            with col:
                st.image(thumbnail(str(path)), width=150)
                if st.button(f"Sample {i}", key=f"sample{i}"):
                    st.session_state.sample = path.name
                    st.rerun()
        if not warmup().ready:
            st.caption("The model is loading in the background (about a minute after the app has been idle), so your first analysis starts sooner.")
        st.caption("Samples are from the Kvasir-VQA-x1 dataset (CC BY-NC 4.0, Simula Research Laboratory), which this model was trained on, "
                   "so they are easier than new images. Upload your own to see how it behaves on unfamiliar ones.")
    else:
        st.info("Choose an image to begin.")
    st.stop()

if upload is not None:
    raw, name = upload.getvalue(), upload.name
else:
    raw, name = (Path(__file__).parent / "samples" / sample).read_bytes(), sample
key = (name, len(raw))
if st.session_state.get("key") != key:  # a new file: forget the last result
    st.session_state.update(key=key, result=None, error=None)

try:
    probe = core.read_image(raw)
    os.unlink(probe[0])
    original = probe[1]
except core.ImageError as e:
    st.error(str(e))
    st.stop()

if st.session_state.get("result") is None:
    left, _ = st.columns([1, 1])
    with left:
        fit(st.image, original, caption="Your image")
    if fit(st.button, "Analyze image", type="primary"):
        path = None
        try:
            with st.spinner("Loading the model if it is not ready yet, then reading the image (this waits if someone else is being analysed)..."):
                pr = predictor()
                path, _ = core.read_image(raw)
                with analysis_lock():
                    t0 = time.perf_counter()
                    result = core.analyse(pr, path)
                    result["pdf"] = core.build_report(path, result)
                    result["seconds"] = time.perf_counter() - t0
            st.session_state.result = result
        except core.ImageError as e:
            st.error(str(e))
        except RuntimeError as e:  # raised by core with a message written for the visitor (weights unavailable, checksum mismatch)
            st.error(str(e))
        except Exception:
            st.error("The model hit an error on that image. Try another image.")
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        if st.session_state.get("result") is not None:
            st.rerun()
    if upload is None and st.button("Choose a different image"):
        fresh_start()
        st.rerun()
    st.stop()

res = st.session_state.result
cam = res["cam"]
col_orig, col_cam = st.columns(2)
bw = st.session_state.get("behind", "Black and white") == "Black and white"  # the choice made with the radio below, kept across runs
with col_orig:
    fit(st.image, original, caption="Original image")
with col_cam:
    if cam and cam.get("overlay"):
        fit(st.image, with_heat(original, cam["overlay"], bw), caption="Grad-CAM overlay: warmer colour means more influence on the finding")
    else:
        st.info("A heatmap was not available for this image.")
if cam and cam.get("overlay"):
    st.radio("Picture behind the heat", ["Black and white", "Original colour"], horizontal=True, key="behind")

head = res["cam_answer"].strip()
head = head[:1].upper() + head[1:]
detail = f'<div class="detail">{res["detail"]}</div>' if res["detail"] else ""
st.markdown(f'<div class="answer"><span class="tag">GastroLens model</span><div class="head">{head}</div>{detail}</div>', unsafe_allow_html=True)

conf = res["confidence"]
if conf.get("abnormality") is not None:
    st.progress(conf["abnormality"] / 100, text=f"Model confidence: {pct(conf['abnormality'])}")
    extra = [f"Polyp present: {pct(conf['polyp'])}"] if conf.get("polyp") is not None else []
    if conf.get("polyp_size") is not None:
        extra.append(f"size band: {pct(conf['polyp_size'])}")
    st.caption(" · ".join(extra + ["The model's own probability, not a guarantee that it is right."]))

filled = [f for f in res["facts"] if f["value"]]
with st.expander(f"Image facts ({len(filled)} of {len(res['facts'])} checks have a value)", expanded=True):
    by_key = {f["key"]: f for f in filled}
    listed = {k for _, ks in res["fact_groups"] for k in ks}
    groups = [(title, [by_key[k] for k in ks if k in by_key]) for title, ks in res["fact_groups"]]
    groups.append(("Other", [f for f in filled if f["key"] not in listed]))
    cols = st.columns(2)
    n = 0
    for title, items in groups:
        if not items:
            continue
        with cols[n % 2]:
            st.markdown(f"**{title}**")
            for f in items:
                st.write(f"{f['label']}: **{display_value(f['value'])}**")
        n += 1

fit(st.download_button, "Download report (PDF)", data=res["pdf"], file_name="GastroLens-report.pdf", mime="application/pdf")
st.caption(f"Analysis took {res['seconds']:.1f} s on this server's CPU.")
if st.button("Analyze another image"):
    fresh_start()
    st.rerun()

with st.expander("About this model"):
    st.write(
        "GastroLens is a 272M-parameter vision-language model (SigLIP2 image encoder, BERT question encoder, 17 fact-classifier heads and a "
        "small answer decoder) trained on the Kvasir-VQA-x1 dataset (CC BY-NC 4.0). The heatmap is a Grad-CAM of the finding shown: it is "
        "relative influence, not a segmentation, and the model was never trained on where things are. Confidence is the model's own "
        "probability and is not calibrated. The model has been trained on the dataset's images and has not been validated by clinicians; "
        "results on new images can be worse than on familiar ones."
    )
