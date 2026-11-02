# GastroLens

A small web demo for a research model that reads GI endoscopy images. Upload an image and the app shows what the model finds, a Grad-CAM heatmap of
where it looked, the model's own confidence, a list of image facts, and a one-page PDF report.

**Research prototype, not a medical device. Not for diagnosis or clinical decisions. Do not upload images of real patients.**

## What is in it

- A 272M-parameter vision-language model: SigLIP2 image encoder, BERT question encoder, 17 fact-classifier heads and a small answer decoder,
  trained on [Kvasir-VQA-x1](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1).
- The heatmap is a Grad-CAM of the finding shown. It is relative influence, not a segmentation.
- Confidence is the model's own probability and is not calibrated.
- The model was trained on the dataset's images and has not been checked by clinicians. Results on new images can be worse than on familiar ones.

## Run it yourself

The weights are not in this repository (1 GB). They are checked against a published SHA-256 before the model runs.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export GASTROLENS_WEIGHTS_REPO=<owner>/gastrolens-weights   # a Hugging Face model repo that holds checkpoints/best.pt and model/*.json
export HF_TOKEN=<read-only token>                           # only if that repo is private
streamlit run app.py
```

Or point `GASTROLENS_WEIGHTS_DIR` at a local folder with `checkpoints/best.pt` and `model/{attr_schema,answer_tokenizer}.json`.
The app runs on CPU (2 threads; about 8 seconds per image) and needs roughly 2 GB of memory.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository and choose `app.py` as the main file (Python 3.12 under *Advanced settings*).
2. Under *Advanced settings, Secrets*, add:
   ```toml
   GASTROLENS_WEIGHTS_REPO = "<owner>/gastrolens-weights"
   HF_TOKEN = "<a read-only token limited to that one repo>"
   ```
3. The first start downloads the weights (about a minute); later starts reuse them.

## Licence and data

The model was trained on a CC BY-NC 4.0 dataset, so the weights and this demo are for **non-commercial research use only**. See `NOTICE.md`.
No dataset images are included. Uploaded images are processed in memory and a temporary file that is deleted straight after.
