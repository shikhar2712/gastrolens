"""Evaluate a checkpoint on the official test split (+ text-only prior baseline, qualitative samples).

python -m kvqa.evaluate --exp experiments/kvasir_vqa_x1_baseline [--ckpt best.pt] [--split test] [--beams 1] [--limit N]
"""
import argparse, json, os, shutil

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

from .data import GroupedVQA
from .facts import fact_summary, fact_table
from .metrics import TEXT_METRICS, attribute_metrics, bootstrap_ci, per_sample, summarize
from .model import KvasirVQAModel
from .train import decode_ids, to_dev


def text_prior_baseline(df_train, df_test):
    """No image: most frequent train answer for the exact question, else for the same class combination."""
    by_q = df_train.groupby("question").answer.agg(lambda s: s.value_counts().index[0])
    key = df_train.question_class.map(tuple)
    by_c = df_train.groupby(key).answer.agg(lambda s: s.value_counts().index[0])
    return [by_q.get(q, by_c.get(tuple(c), "")) for q, c in zip(df_test.question, df_test.question_class)]


def md_table(d, cols):
    lines = ["| group | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k, v in d.items():
        lines.append(f"| {k} | {int(v['n'])} | " + " | ".join(f"{v[c]:.4f}" for c in cols) + " |")
    return "\n".join(lines)


def pick_samples(df, rng, raw_dir, out_dir, n_per=5):
    buckets = {
        "correct (exact match)": df[df.exact_match == 1],
        "partially correct (0.4 <= ROUGE-L < 0.9, not exact)": df[(df.exact_match == 0) & df.rougeL.between(0.4, 0.9, inclusive="left")],
        "clearly incorrect (ROUGE-L < 0.3)": df[df.rougeL < 0.3],
        "difficult: complexity 3": df[df.complexity == 3],
        "unseen test images": df[~df.seen_image],
    }
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    md = ["# Qualitative test samples (random within each bucket, seed 0 — not cherry-picked)\n"]
    for name, sub in buckets.items():
        md.append(f"\n## {name}  (bucket size {len(sub)})\n")
        for _, r in sub.sample(min(n_per, len(sub)), random_state=int(rng.integers(1e9))).iterrows():
            shutil.copy(f"{raw_dir}/images/{r.img_id}.jpg", os.path.join(out_dir, "images", f"{r.img_id}.jpg"))
            md.append(f"![](images/{r.img_id}.jpg)\n\n- **Q** (c{r.complexity}; {', '.join(r.question_class)}; "
                      f"{'seen' if r.seen_image else 'unseen'} image): {r.question}\n- **GT**: {r.answer}\n"
                      f"- **Pred**: {r.prediction}\n- ROUGE-L {r.rougeL:.3f} | METEOR {r.meteor:.3f} | EM {int(r.exact_match)}\n")
    open(os.path.join(out_dir, "samples.md"), "w").write("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beams", type=int, default=1, help="1 = greedy")
    ap.add_argument("--length_penalty", type=float, default=1.0)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.exp, "checkpoints", args.ckpt), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cache = cfg["data"]["cache_dir"]
    atok = Tokenizer.from_file(f"{cache}/answer_tokenizer.json")
    ds = GroupedVQA(cfg, args.split, train=False)
    ds.groups = ds.make_groups(None, k=32)
    if args.limit:
        ds.groups = ds.groups[: args.limit]
    model = KvasirVQAModel(cfg, ds.attr.schema)
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}, step {ck['step']}); {args.split} groups {len(ds.groups)}", flush=True)

    preds, rows, attr_logits = [], [], {}
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["eval"]["batch_questions"] // 32, collate_fn=ds.collate,
                                     num_workers=cfg["data"]["num_workers"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, b in enumerate(dl):
            b = to_dev(b, "cuda")
            if args.beams > 1:
                ids, al = model.generate_beam(b, cfg["eval"]["max_gen_len"], args.beams, args.length_penalty)
            else:
                ids, al = model.generate(b, cfg["eval"]["max_gen_len"])
            preds += decode_ids(atok, ids)
            rows += b["rows"]
            for gi, img in enumerate(b["img_ids"]):
                attr_logits[img] = {c: al[c][gi].float().cpu().numpy() for c in al}
            if i % 20 == 0:
                print(f"  batch {i}/{len(dl)}", flush=True)

    df = ds.df.iloc[rows].copy()
    df["prediction"] = preds
    m = per_sample(preds, df.answer.tolist())
    for c in TEXT_METRICS:
        df[c] = m[c].values
    tag = args.split + (f"_beam{args.beams}" if args.beams > 1 else "") + (f"_limit{args.limit}" if args.limit else "")
    for sub in ["predictions", "metrics", "samples"]:
        os.makedirs(os.path.join(args.exp, sub), exist_ok=True)
    cols = ["orig_idx", "img_id", "complexity", "question_class", "seen_image", "question", "answer", "prediction"] + TEXT_METRICS
    if "seen_group" in df:
        cols.insert(5, "seen_group")
    df[cols].to_json(os.path.join(args.exp, "predictions", f"{tag}_predictions.jsonl"), orient="records", lines=True)

    res = summarize(df, preds, df.answer.tolist())
    res["checkpoint"] = {"file": args.ckpt, "epoch": ck["epoch"], "step": ck["step"], "val": ck["val"]}
    res["decoding"] = {"beams": args.beams, "length_penalty": args.length_penalty}
    res["ci95_image_bootstrap"] = bootstrap_ci(df, TEXT_METRICS)
    ft = fact_table(df[df.complexity == 1])
    res["facts_complexity1"] = fact_summary(ft)
    ck_ft = ft[ft.checkable].assign(correct=lambda x: x.correct.astype(float))
    res["facts_complexity1"]["overall"]["ci95"] = bootstrap_ci(ck_ft, ["correct"])["correct"]
    seen_imgs = set(ds.df.img_id[ds.df.seen_image])
    res["attribute_heads"] = attribute_metrics(ds.attr, ds.facts, attr_logits, seen_imgs)

    # text-only prior baseline on the same rows
    full = pd.read_parquet(f"{cache}/splits.parquet")
    bp = text_prior_baseline(full[full.split == "train"], df)
    bm = per_sample(bp, df.answer.tolist())
    meta = [c for c in ["question_class", "complexity", "seen_image", "seen_group"] if c in df]
    res["text_prior_baseline"] = summarize(df[meta].assign(**{c: bm[c].values for c in TEXT_METRICS}), bp, df.answer.tolist())
    res["text_prior_baseline"]["facts_complexity1"] = fact_summary(fact_table(df.assign(prediction=bp)[df.complexity == 1]))
    json.dump(res, open(os.path.join(args.exp, "metrics", f"{tag}_metrics.json"), "w"), indent=1, default=float)

    ov_cols = ["exact_match", "bleu_corpus", "bleu_sent", "rouge1", "rouge2", "rougeL", "meteor"]
    md = [f"# {tag} metrics — checkpoint {args.ckpt} (epoch {ck['epoch']})\n",
          md_table({"model": res["overall"], "text-only prior baseline": res["text_prior_baseline"]["overall"]}, ov_cols)]
    for k in ["by_complexity", "by_seen_image", "by_seen_group", "by_answer_type_complexity1", "by_question_class"]:
        if k not in res:
            continue
        md += [f"\n## {k}\n", md_table(res[k], TEXT_METRICS)]
    md += ["\n## text-only prior baseline by_question_class\n", md_table(res["text_prior_baseline"]["by_question_class"], TEXT_METRICS)]
    fo, fb = res["facts_complexity1"], res["text_prior_baseline"]["facts_complexity1"]
    ci = res["ci95_image_bootstrap"]
    md += [f"\n## 95% CI (image bootstrap)\n\nROUGE-L [{ci['rougeL'][0]:.4f}, {ci['rougeL'][1]:.4f}] | METEOR "
           f"[{ci['meteor'][0]:.4f}, {ci['meteor'][1]:.4f}] | EM [{ci['exact_match'][0]:.4f}, {ci['exact_match'][1]:.4f}]\n",
           f"\n## clinical fact accuracy, complexity-1 rows (kvqa.facts)\n\nmodel {fo['overall']['fact_acc']:.4f} "
           f"[{fo['overall']['ci95'][0]:.4f}, {fo['overall']['ci95'][1]:.4f}] (macro {fo['overall']['macro_fact_acc']:.4f}) | "
           f"text-only prior {fb['overall']['fact_acc']:.4f} (macro {fb['overall']['macro_fact_acc']:.4f}) | "
           f"n checkable {fo['overall']['n']} (coverage {fo['overall']['coverage']:.3f})\n",
           "| class | n | fact_acc | prior | coverage |", "|---|---|---|---|---|"]
    md += [f"| {c} | {int(v['n'])} | {v['fact_acc']:.4f} | {fb['by_class'][c]['fact_acc']:.4f} | {v['coverage']:.3f} |"
           for c, v in fo["by_class"].items()]
    ah = res["attribute_heads"]
    md += [f"\n## atomic attribute heads (image level)\n\noverall acc {ah['overall']['acc']:.4f} f1 {ah['overall']['f1']:.4f} (n={ah['overall']['n']})\n",
           md_table(ah["by_seen"], ["acc", "f1"]), "", md_table(ah["by_cls"], ["acc", "f1"])]
    open(os.path.join(args.exp, "metrics", f"{tag}_metrics.md"), "w").write("\n".join(md))
    if not args.limit:
        pick_samples(df, np.random.default_rng(0), cfg["data"]["raw_dir"], os.path.join(args.exp, "samples", tag))
    print("\n".join(md[:2]))
    print(f"fact acc (c1) {fo['overall']['fact_acc']:.4f} {fo['overall']['ci95']} macro {fo['overall']['macro_fact_acc']:.4f}")


if __name__ == "__main__":
    main()
