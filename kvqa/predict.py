"""Answer questions about new endoscopy images with a trained checkpoint (+ the answer-consistency step).

  python -m kvqa.predict --exp experiments/kvasir_vqa_x1_final --image path/to/img.jpg \
      --question "What is the size of the polyp?" [--question "..."] [--no-consistency]
  python -m kvqa.predict --exp experiments/kvasir_vqa_x1_final --jsonl requests.jsonl --out answers.jsonl
      (each request line: {"image": "<path>", "question": "<text>"})

Preprocessing matches training exactly (kvqa.prepare + GroupedVQA.collate): RGB, bicubic resize to image_size x image_size,
pixel/255, then (x - 0.5) / 0.5. Output per question: the final answer, the plain greedy answer, whether the consistency
step changed it, the predicted question class, and the model's image-level fact-classifier outputs.
"""
import argparse, json, os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .consistency import DEFAULT_ROUTE, RECOMMENDED_ROUTE, QuestionClassifier, consistent_answer, head_labels
from .data import AttrEncoder
from .model import KvasirVQAModel
from .train import decode_ids


def load_image(path, res):
    """uint8 CHW tensor, identical to the training cache (kvqa.prepare: RGB, bicubic resize to res x res)."""
    im = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


class Predictor:
    def __init__(self, exp, ckpt="best.pt", device="cuda", route=DEFAULT_ROUTE, beams=8):
        lean = os.environ.get("GASTROLENS_LEAN") == "1"  # memory-lean start-up for small hosts (see model.py)
        ck = torch.load(os.path.join(exp, "checkpoints", ckpt), map_location="cpu", weights_only=False, mmap=lean)
        self.cfg = cfg = ck["config"]
        d = cfg["data"]
        model_dir = os.path.join(exp, "model")
        schema_path = os.path.join(model_dir, "attr_schema.json")
        tok_path = os.path.join(model_dir, "answer_tokenizer.json")
        if not os.path.exists(schema_path):
            schema_path, tok_path = f"{d['cache_dir']}/attr_schema.json", f"{d['cache_dir']}/answer_tokenizer.json"
        self.attr = AttrEncoder(json.load(open(schema_path)))
        self.atok = Tokenizer.from_file(tok_path)
        self.qtok = AutoTokenizer.from_pretrained(cfg["model"]["text"])
        if lean:
            # Build the model with no memory (meta device) and let it use the memory-mapped checkpoint tensors directly: no second copy of
            # the 1 GB of weights is ever held, so start-up peaks near the final size instead of at twice it.
            with torch.device("meta"):
                self.model = KvasirVQAModel(cfg, self.attr.schema)
            self.model.load_state_dict(ck["model"], assign=True)
            for name, buf in list(self.model.named_buffers()):  # the few non-persistent index buffers are not in the checkpoint
                if buf.is_meta:
                    if "position_ids" in name:
                        value = torch.arange(buf.shape[-1], dtype=buf.dtype).expand(buf.shape).clone()
                    elif "token_type_ids" in name:
                        value = torch.zeros(buf.shape, dtype=buf.dtype)
                    else:
                        raise RuntimeError(f"buffer {name} has no value after loading the checkpoint")
                    owner = self.model
                    *path, leaf = name.split(".")
                    for part in path:
                        owner = getattr(owner, part)
                    owner.register_buffer(leaf, value, persistent=False)
            del ck
        else:
            self.model = KvasirVQAModel(cfg, self.attr.schema)
            self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device, self.route, self.beams = device, set(route or ()), beams
        self.res = d["image_size"]
        if self.route:
            tr = pd.read_parquet(f"{d['cache_dir']}/splits.parquet", columns=["split", "question", "question_class"])
            tr = tr[tr.split == "train"]
            self.qc = QuestionClassifier(tr.question.tolist(), [list(c) for c in tr.question_class])

    def load_image(self, path):
        return load_image(path, self.res)

    @torch.no_grad()
    def answer(self, requests):
        """requests: list of (image_path, question). Images are encoded once per distinct path."""
        paths = list(OrderedDict.fromkeys(p for p, _ in requests))
        idx = {p: i for i, p in enumerate(paths)}
        images = torch.stack([self.load_image(p) for p in paths]).float().div(255).sub(0.5).div(0.5)
        q = self.qtok([qq for _, qq in requests], padding=True, truncation=True,
                      max_length=self.cfg["data"]["max_question_len"], return_tensors="pt")
        batch = {"images": images.to(self.device), "img_index": torch.tensor([idx[p] for p, _ in requests], device=self.device),
                 "q_ids": q.input_ids.to(self.device), "q_mask": q.attention_mask.to(self.device)}
        max_len = self.cfg["eval"]["max_gen_len"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            greedy_ids, logits = self.model.generate(batch, max_len)
            greedy = decode_ids(self.atok, greedy_ids)
            beams = None
            if self.route:
                _, _, allseq, _ = self.model.generate_beam(batch, max_len, self.beams, return_all=True)
                beams = [decode_ids(self.atok, allseq[:, k]) for k in range(self.beams)]
        heads = [head_labels(self, logits, gi) for gi in range(len(paths))]
        out = []
        for j, (p, qq) in enumerate(requests):
            h = heads[idx[p]]
            final, changed, cls = greedy[j], False, None
            if self.route:
                cls = list(self.qc.predict(qq))
                if len(cls) == 1 and cls[0] in self.route:
                    final, changed = consistent_answer(cls[0], h[cls[0]], greedy[j], [b[j] for b in beams])
            out.append({"image": p, "question": qq, "answer": final, "greedy_answer": greedy[j],
                        "consistency_changed": changed, "predicted_question_class": cls, "fact_classifiers": h})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--image")
    ap.add_argument("--question", action="append")
    ap.add_argument("--jsonl", help="file of {image, question} lines")
    ap.add_argument("--out", help="write answers as JSONL (default: print)")
    ap.add_argument("--no-consistency", action="store_true")
    ap.add_argument("--route", default=",".join(DEFAULT_ROUTE),
                    help="comma-separated fact types the consistency step corrects. Default = the released setting "
                         f"({','.join(DEFAULT_ROUTE)}); recommended for new use: {','.join(RECOMMENDED_ROUTE)}")
    ap.add_argument("--batch", type=int, default=64, help="questions per forward pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.jsonl:
        reqs = [(r["image"], r["question"]) for r in map(json.loads, open(a.jsonl))]
    else:
        assert a.image and a.question, "give --image and --question, or --jsonl"
        reqs = [(a.image, q) for q in a.question]
    pr = Predictor(a.exp, a.ckpt, a.device, route=None if a.no_consistency else [r for r in a.route.split(",") if r])
    res = []
    for i in range(0, len(reqs), a.batch):
        res += pr.answer(reqs[i:i + a.batch])
    if a.out:
        with open(a.out, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(res)} answers to {a.out}")
    else:
        for r in res:
            print(json.dumps({k: r[k] for k in ("question", "answer", "consistency_changed")}))


if __name__ == "__main__":
    main()
