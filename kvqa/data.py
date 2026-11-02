"""Image-grouped dataset: each item is one image + up to K of its questions."""
import json

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from tokenizers import Tokenizer
from transformers import AutoTokenizer

PAD, BOS, EOS = 0, 1, 2


class AttrEncoder:
    """Turns per-image atomic facts into head targets + masks."""

    def __init__(self, schema):
        # heads with a single label carry no information
        self.schema = {c: s for c, s in schema.items() if len(s["labels"]) >= 2}
        self.classes = sorted(self.schema)
        self.index = {c: {l: i for i, l in enumerate(s["labels"])} for c, s in self.schema.items()}

    def encode(self, facts):
        """facts: {class: raw answer string} -> {class: (target, mask)}; unknown single labels -> -1."""
        out = {}
        for c in self.classes:
            s, idx = self.schema[c], self.index[c]
            if s["type"] == "multi":
                t = np.zeros(len(s["labels"]), np.float32)
                if c in facts:
                    for l in filter(None, facts[c].split(";")):
                        if l in idx:
                            t[idx[l]] = 1
                out[c] = (t, float(c in facts))
            else:
                out[c] = (idx.get(facts.get(c), -1), float(c in facts and facts[c] in idx))
        return out


class GroupedVQA(torch.utils.data.Dataset):
    def __init__(self, cfg, split, train, groups=None):
        d = cfg["data"]
        cache = d["cache_dir"]
        df = pd.read_parquet(f"{cache}/splits.parquet")
        self.df = df[df.split == split].reset_index(drop=True)
        ids = json.load(open(f"{cache}/image_ids.json"))
        self.img_row = {i: k for k, i in enumerate(ids)}
        self.images = np.load(f"{cache}/images_{d['image_size']}.npy", mmap_mode="r")
        facts = json.load(open(f"{cache}/image_attrs.json"))[split]
        self.attr = AttrEncoder(json.load(open(f"{cache}/attr_schema.json")))
        self.facts = facts
        self.train, self.cfg, self.k = train, cfg, d["questions_per_image"]
        self.qtok = AutoTokenizer.from_pretrained(cfg["model"]["text"])
        self.atok = Tokenizer.from_file(f"{cache}/answer_tokenizer.json")
        a = d["augment"]
        self.aug = T.Compose([
            T.RandomResizedCrop(d["image_size"], scale=tuple(a["crop_scale"]), ratio=(0.9, 1.1), antialias=True),
            T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        ])
        self.groups = groups if groups is not None else self.make_groups(np.random.default_rng(cfg["seed"]))

    def make_groups(self, rng, k=None):
        """Shuffle rows within each image and chunk into groups of <= k questions."""
        k = k or self.k
        groups = []
        for _, idx in self.df.groupby("img_id").indices.items():
            idx = rng.permutation(idx) if self.train else np.sort(idx)
            groups += [idx[i:i + k] for i in range(0, len(idx), k)]
        if self.train:
            groups = [groups[i] for i in rng.permutation(len(groups))]
        return groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, g):
        rows = self.df.iloc[self.groups[g]]
        img_id = rows.img_id.iloc[0]
        img = torch.from_numpy(np.array(self.images[self.img_row[img_id]])).permute(2, 0, 1)
        if self.train:
            img = self.aug(img)
        return dict(image=img, img_id=img_id, rows=self.groups[g], questions=rows.question.tolist(),
                    answers=rows.answer.tolist(), attrs=self.attr.encode(self.facts.get(img_id, {})))

    def collate(self, batch):
        d = self.cfg["data"]
        images = torch.stack([b["image"] for b in batch]).float().div(255).sub(0.5).div(0.5)
        img_index = torch.tensor([gi for gi, b in enumerate(batch) for _ in b["questions"]])
        questions = [q for b in batch for q in b["questions"]]
        answers = [a for b in batch for a in b["answers"]]
        q = self.qtok(questions, padding=True, truncation=True, max_length=d["max_question_len"], return_tensors="pt")
        L = d["max_answer_len"]
        enc = [[BOS] + e.ids[:L - 2] + [EOS] for e in self.atok.encode_batch(answers)]
        ans = torch.full((len(enc), max(map(len, enc))), PAD, dtype=torch.long)
        for i, e in enumerate(enc):
            ans[i, :len(e)] = torch.tensor(e)
        attr_t, attr_m = {}, {}
        for c in self.attr.classes:
            ts = [b["attrs"][c][0] for b in batch]
            attr_t[c] = torch.tensor(np.stack(ts)) if self.attr.schema[c]["type"] == "multi" else torch.tensor(ts)
            attr_m[c] = torch.tensor([b["attrs"][c][1] for b in batch])
        return dict(images=images, img_index=img_index, q_ids=q.input_ids, q_mask=q.attention_mask, ans=ans,
                    attr_t=attr_t, attr_m=attr_m, rows=[r for b in batch for r in b["rows"]],
                    img_ids=[b["img_id"] for b in batch])
