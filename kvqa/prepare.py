"""One-off preprocessing for Kvasir-VQA-x1.

Produces (in cfg.data.cache_dir):
  images_<res>.npy     uint8 (N, res, res, 3) resized images, row order = image_ids.json
  image_ids.json       list of img_id
  splits.parquet       all rows with a `split` column in {train, val, test}
  attr_schema.json     atomic-attribute label spaces (built from train split only)
  image_attrs.json     {split_group: {img_id: {question_class: raw_answer}}} per-image atomic facts
  answer_tokenizer.json  byte-level BPE tokenizer for the answer decoder (trained on train answers only)
"""
import argparse, collections, json, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Atomic question classes that allow multiple simultaneous values ("a; b; c").
MULTI_LABEL = {
    "abnormality_presence", "polyp_type", "landmark_presence", "instrument_presence", "polyp_size",
    "abnormality_color", "landmark_color", "abnormality_location", "instrument_location", "landmark_location",
}
# Typos / inconsistent spellings present in the raw atomic answers.
FIXES = {"lower-rigth": "lower-right", ">20": ">20mm"}


def split_multi(ans):
    toks = [FIXES.get(t.strip().lower(), t.strip().lower()) for t in ans.split(";")]
    return sorted({t for t in toks if t and t != "none"})


def norm_single(ans):
    return ans.strip().lower()


def per_image_facts(df):
    """Majority vote per (image, atomic class) over all rows that mention it."""
    votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for img, orig, classes in zip(df.img_id, df.original, df.question_class):
        for qa, c in zip(json.loads(orig), classes):
            key = ";".join(split_multi(qa["a"])) if c in MULTI_LABEL else norm_single(qa["a"])
            votes[img][c][key] += 1
    return {img: {c: cnt.most_common(1)[0][0] for c, cnt in cs.items()} for img, cs in votes.items()}


def image_stratum(facts):
    """Coarse clinical stratum of an image: (procedure, abnormality set, instrument present)."""
    inst = facts.get("instrument_presence")
    return (facts.get("procedure_type", "no-finding"), facts.get("abnormality_presence", "-"),
            "instrument" if inst else "-")


def select_heldout(tr, groups, d, seed, frac=None, seed_offset=1):
    """Pick ~frac (default heldout_frac) of train images, as whole near-duplicate groups, stratified by clinical stratum."""
    frac = d["heldout_frac"] if frac is None else frac
    facts = per_image_facts(tr)
    imgs = sorted(tr.img_id.unique())
    group_size = collections.Counter(groups.values())  # over all images, incl. test-only ones
    by_group = collections.defaultdict(list)
    for i in imgs:
        by_group[groups[i]].append(i)
    eligible = {g: m for g, m in by_group.items() if group_size[g] <= d["max_group_size"]}
    strata = collections.defaultdict(list)
    for g, members in eligible.items():
        key = collections.Counter(image_stratum(facts[i]) for i in members).most_common(1)[0][0]
        strata[key].append(g)
    n_per_stratum = collections.Counter(image_stratum(facts[i]) for i in imgs)
    rng = np.random.default_rng(seed + seed_offset)
    held = set()
    for key in sorted(strata):
        target, taken = round(frac * n_per_stratum[key]), 0
        for gi in rng.permutation(len(strata[key])):
            if taken >= target:
                break
            members = eligible[strata[key][gi]]
            held.update(members)
            taken += len(members)
    return held


def main(cfg):
    d = cfg["data"]
    raw, out = d["raw_dir"], d["cache_dir"]
    os.makedirs(out, exist_ok=True)
    tr = pd.read_parquet(f"{raw}/data/train-00000-of-00001.parquet")
    te = pd.read_parquet(f"{raw}/data/test-00000-of-00001.parquet")
    tr["orig_idx"], te["orig_idx"] = np.arange(len(tr)), np.arange(len(te))

    # ---- optional image-disjoint held-out split carved from the official TRAIN split ----
    groups = json.load(open(d["dup_groups"])) if d.get("dup_groups") else None
    if d.get("split_mode", "official") == "image_disjoint":
        held = select_heldout(tr, groups, d, cfg["seed"])
        tr["split"] = np.where(tr.img_id.isin(held), "heldout", "train")
        ho = tr[tr.split == "heldout"].reset_index(drop=True)
        tr = tr[tr.split == "train"].reset_index(drop=True)
        print(f"held out {len(held)} images / {len(ho)} rows (whole near-duplicate groups)")
    else:
        ho = tr.iloc[:0].copy()

    # ---- validation split, from the official TRAIN split only; test untouched ----
    rng = np.random.default_rng(cfg["seed"])
    if d.get("val_mode", "rows") == "groups":
        # whole near-duplicate groups, so checkpoint selection measures performance on new images
        val_imgs = select_heldout(tr, groups, d, cfg["seed"], frac=d["val_frac"], seed_offset=2)
        is_val = tr.img_id.isin(val_imgs).to_numpy()
        print(f"validation: {len(val_imgs)} images / {int(is_val.sum())} rows (whole near-duplicate groups)")
    else:  # random rows (the baseline's choice): val images are also training images
        is_val = rng.random(len(tr)) < d["val_frac"]
    tr["split"] = np.where(is_val, "val", "train")
    fit_pairs = set(zip(tr.img_id[~is_val], tr.question[~is_val]))
    # drop val rows whose exact (image, question) also occurs in the fitting set (trivial duplicates)
    dup = tr.split.eq("val").to_numpy() & np.array([p in fit_pairs for p in zip(tr.img_id, tr.question)])
    tr = tr[~dup].copy()
    te["split"] = "test"
    df = pd.concat([tr, ho, te], ignore_index=True)
    df["question_class"] = df.question_class.map(list)
    train_imgs = set(tr.img_id[tr.split == "train"])
    df["seen_image"] = df.img_id.isin(train_imgs)
    if groups is not None:  # seen if any near-duplicate of the image was trained on
        train_groups = {groups[i] for i in train_imgs}
        df["seen_group"] = df.img_id.map(lambda i: groups[i] in train_groups)
    df.to_parquet(f"{out}/splits.parquet")
    print("rows per split:", df.split.value_counts().to_dict(), "| val rows dropped as exact dups:", int(dup.sum()))
    for s in ["test", "heldout"]:
        sub = df[df.split == s]
        if len(sub):
            msg = f"{s} rows on images unseen in training: {int((~sub.seen_image).sum())}"
            if "seen_group" in df:
                msg += f" | with no near-duplicate in training: {int((~sub.seen_group).sum())}"
            print(msg)

    # ---- atomic-attribute schema from train split only ----
    fit = df[df.split == "train"]
    facts = {s: per_image_facts(df[df.split == s]) for s in df.split.unique()}
    counts = collections.defaultdict(collections.Counter)
    for img_facts in facts["train"].values():
        for c, a in img_facts.items():
            for lab in (a.split(";") if c in MULTI_LABEL else [a]):
                if lab:
                    counts[c][lab] += 1
    schema = {}
    for c in sorted(counts):
        labs = [l for l, n in counts[c].most_common() if n >= d["min_label_count"]]
        schema[c] = {"type": "multi" if c in MULTI_LABEL else "single", "labels": sorted(labs)}
    json.dump(schema, open(f"{out}/attr_schema.json", "w"), indent=1)
    json.dump(facts, open(f"{out}/image_attrs.json", "w"))
    print("attribute heads:", {c: (s["type"], len(s["labels"])) for c, s in schema.items()})

    # ---- answer tokenizer (byte-level BPE, reversible) on train answers only ----
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=d["answer_vocab_size"], min_frequency=2,
                                  special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(fit.answer.tolist(), trainer)
    tok.save(f"{out}/answer_tokenizer.json")
    lens = [len(e.ids) for e in tok.encode_batch(df.answer.tolist())]
    rt = np.mean([tok.decode(tok.encode(a).ids) == a for a in df.answer.sample(2000, random_state=0)])
    print(f"answer vocab {tok.get_vocab_size()} | tokens/answer max {max(lens)} p99 {np.percentile(lens, 99):.0f} | round-trip exact {rt:.3f}")

    # ---- image cache ----
    res = d["image_size"]
    ids = sorted(df.img_id.unique())
    if d.get("reuse_images_from"):  # same images, same resize: link instead of re-encoding
        src = d["reuse_images_from"]
        assert set(ids) <= set(json.load(open(f"{src}/image_ids.json"))), "reused cache is missing images"
        for f in [f"images_{res}.npy", "image_ids.json"]:
            dst = f"{out}/{f}"
            if not os.path.exists(dst):
                if os.path.islink(dst):  # stale link (e.g. the project folder was moved): replace it
                    os.remove(dst)
                # relative link, so the project folder can be moved or renamed
                os.symlink(os.path.relpath(f"{src}/{f}", out), dst)
        print(f"linked image cache from {src}")
        return
    arr = np.lib.format.open_memmap(f"{out}/images_{res}.npy", mode="w+", dtype=np.uint8, shape=(len(ids), res, res, 3))

    def load(i):
        arr[i] = np.asarray(Image.open(f"{raw}/images/{ids[i]}.jpg").convert("RGB").resize((res, res), Image.BICUBIC))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, range(len(ids))))
    arr.flush()
    json.dump(ids, open(f"{out}/image_ids.json", "w"))
    print(f"cached {len(ids)} images at {res}x{res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
