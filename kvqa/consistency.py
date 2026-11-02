"""Answer/fact-head consistency: for single-fact questions of classes where the model's image-level fact heads are
more reliable than its free text, pick the best beam whose stated fact agrees with the head.

Inference uses only the image and question text: the question's class is predicted from the text (exact lookup
of training questions, naive-Bayes fallback), never read from the dataset's question_class column.
Which classes to route is chosen by 2-fold cross-fitting over images (choose on one half, apply to the other),
so the reported gain is not tuned on the rows it is measured on.

python -m kvqa.consistency --exp experiments/kvasir_vqa_x1_imgdisjoint --split heldout [--beams 8] [--route a,b]

--route fixes the routed classes instead of cross-fitting. The released results use abnormality_location,polyp_type
(DEFAULT_ROUTE: +0.43 to +0.48 label-match points per seed on our own checker). The independent LLM judge does not confirm
that gain, so for new use we recommend --route abnormality_location (RECOMMENDED_ROUTE).
"""
import argparse, collections, hashlib, json, math, os, re

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

from .data import GroupedVQA
from .facts import canonical, extract, fact_summary, fact_table
from .metrics import TEXT_METRICS, bootstrap_ci, per_sample, summarize
from .model import KvasirVQAModel
from .train import decode_ids, to_dev


# DEFAULT_ROUTE: what the released answers and all published numbers use (the only classes chosen in both
# cross-fitting folds for all 3 image-disjoint seeds; experiments/analyses/seed_noise_2026-09-28.md).
# RECOMMENDED_ROUTE: what we recommend for new use. The LLM judge found the polyp-type correction slightly harmful and the
# location correction slightly helpful (experiments/analyses/judge_calibration/consistency_step_judged.json). It is not
# yet applied to the released numbers: switching requires re-running the consistency and judge stages.
DEFAULT_ROUTE = ("abnormality_location", "polyp_type")
RECOMMENDED_ROUTE = ("abnormality_location",)


# ---------------- question-class prediction (text only) ----------------
def _feats(q):
    w = re.sub(r"[^a-z0-9\s]", " ", q.lower()).split()
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


class QuestionClassifier:
    def __init__(self, questions, classes):
        votes = collections.defaultdict(collections.Counter)
        for q, c in zip(questions, classes):
            votes[q][tuple(sorted(c))] += 1
        self.lookup = {q: v.most_common(1)[0][0] for q, v in votes.items()}
        self.prior, self.count, self.total = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
        vocab = set()
        for q, c in zip(questions, classes):
            c = tuple(sorted(c))
            self.prior[c] += 1
            for f in _feats(q):
                self.count[c][f] += 1; self.total[c] += 1; vocab.add(f)
        self.V, self.N = len(vocab), sum(self.prior.values())

    def predict(self, q):
        if q in self.lookup:
            return self.lookup[q]
        fs = _feats(q)
        best, bs = None, -math.inf
        for c, n in self.prior.items():
            s = math.log(n / self.N) + sum(math.log((self.count[c][f] + 1) / (self.total[c] + self.V)) for f in fs)
            if s > bs:
                best, bs = c, s
        return best


# ---------------- candidates: greedy + beams + head predictions ----------------
def head_labels(ds, logits, gi):
    out = {}
    for c, lg in logits.items():
        s, v = ds.attr.schema[c], lg[gi].float()
        if s["type"] == "multi":
            out[c] = ";".join(sorted(l for l, x in zip(s["labels"], (v > 0).tolist()) if x))
        else:
            out[c] = s["labels"][int(v.argmax())]
    return out


@torch.no_grad()
def generate_candidates(exp, split, beams, path, limit=None):
    ck = torch.load(os.path.join(exp, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    atok = Tokenizer.from_file(f"{cfg['data']['cache_dir']}/answer_tokenizer.json")
    ds = GroupedVQA(cfg, split, train=False)
    ds.groups = ds.make_groups(None, k=32)
    if limit:
        ds.groups = ds.groups[:limit]
    model = KvasirVQAModel(cfg, ds.attr.schema)
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["eval"]["batch_questions"] // 32 // 2, collate_fn=ds.collate,
                                     num_workers=cfg["data"]["num_workers"])
    recs, heads = [], {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, b in enumerate(dl):
            b = to_dev(b, "cuda")
            greedy, logits = model.generate(b, cfg["eval"]["max_gen_len"])
            _, _, allseq, scores = model.generate_beam(b, cfg["eval"]["max_gen_len"], beams, return_all=True)
            g = decode_ids(atok, greedy)
            bt = [decode_ids(atok, allseq[:, k]) for k in range(beams)]
            for gi, img in enumerate(b["img_ids"]):
                heads[img] = head_labels(ds, logits, gi)
            for j, r in enumerate(b["rows"]):
                recs.append(dict(row=int(r), greedy=g[j], beams=[bt[k][j] for k in range(beams)],
                                 scores=scores[j].float().tolist()))
            if i % 20 == 0:
                print(f"  batch {i}/{len(dl)}", flush=True)
    df = ds.df.iloc[[r["row"] for r in recs]].reset_index(drop=True)
    out = pd.DataFrame(recs).drop(columns="row")
    out = pd.concat([df[["orig_idx", "img_id", "complexity", "question_class", "seen_image", "question", "answer", "original"]], out], axis=1)
    out["heads"] = out.img_id.map(lambda i: json.dumps(heads[i]))
    out.to_json(path, orient="records", lines=True)
    return out, cfg


# ---------------- selection ----------------
def consistent_answer(cls, head_raw, greedy, beams):
    h = canonical(cls, head_raw)
    if extract(cls, greedy) == h:
        return greedy, False
    for b in beams:  # best first
        if extract(cls, b) == h:
            return b, True
    return greedy, False


def apply(df, route):
    preds, changed = [], []
    for r in df.itertuples():
        c = r.pred_cls
        if len(c) == 1 and c[0] in route:
            p, ch = consistent_answer(c[0], json.loads(r.heads)[c[0]], r.greedy, r.beams)
        else:
            p, ch = r.greedy, False
        preds.append(p); changed.append(ch)
    return preds, changed


def fact_acc_by_class(df, preds):
    ft = fact_table(df.assign(prediction=preds)[df.complexity == 1])
    ck = ft[ft.checkable]
    return ck.groupby("cls").correct.agg(["sum", "size"])


def choose_route(df, classes):
    """Route a class if routing raises its fact accuracy on these rows (strictly)."""
    base = fact_acc_by_class(df, df.greedy.tolist())
    route = set()
    for c in classes:
        p, _ = apply(df, {c})
        a = fact_acc_by_class(df, p)
        if c in a.index and c in base.index and a.loc[c, "sum"] > base.loc[c, "sum"]:
            route.add(c)
    return route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--beams", type=int, default=8)
    ap.add_argument("--route", default=None, help="comma-separated classes to route (skip cross-fitting)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N 32-question groups (smoke tests)")
    ap.add_argument("--regen", action="store_true", help="regenerate candidates even if a matching cache exists")
    args = ap.parse_args()
    out_dir = os.path.join(args.exp, "consistency")
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, f"{args.split}_candidates_beam{args.beams}" + (f"_limit{args.limit}" if args.limit else "") + ".jsonl")
    ck_path = os.path.join(args.exp, "checkpoints", "best.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ident = {"ckpt_size": os.path.getsize(ck_path), "ckpt_mtime": os.path.getmtime(ck_path), "epoch": ck["epoch"],
             "step": ck["step"], "beams": args.beams, "limit": args.limit}
    meta_path = cand_path + ".meta.json"
    cached = os.path.exists(cand_path) and os.path.exists(meta_path) and json.load(open(meta_path)) == ident
    if cached and not args.regen:
        df, cfg = pd.read_json(cand_path, lines=True), ck["config"]
        print(f"reusing candidates {cand_path} (same checkpoint)")
    else:
        df, cfg = generate_candidates(args.exp, args.split, args.beams, cand_path, args.limit)
        json.dump(ident, open(meta_path, "w"))
    del ck
    df["question_class"] = df.question_class.map(list)

    full = pd.read_parquet(f"{cfg['data']['cache_dir']}/splits.parquet", columns=["split", "question", "question_class"])
    tr = full[full.split == "train"]
    qc = QuestionClassifier(tr.question.tolist(), [list(c) for c in tr.question_class])
    df["pred_cls"] = df.question.map(qc.predict)
    qacc = float(np.mean([tuple(sorted(a)) == b for a, b in zip(df.question_class, df.pred_cls)]))
    c1 = df.complexity == 1
    qacc1 = float(np.mean([tuple(sorted(a)) == b for a, b in zip(df.question_class[c1], df.pred_cls[c1])]))

    classes = sorted({c for c in json.loads(df.heads.iloc[0])} - {"finding_presence", "polyp_removal_status"})
    fold = df.img_id.map(lambda i: int(hashlib.md5(i.encode()).hexdigest(), 16) % 2)
    preds, changed = df.greedy.tolist(), [False] * len(df)
    routes = {}
    for k in (0, 1):  # choose on fold 1-k, apply to fold k
        route = set(args.route.split(",")) if args.route else choose_route(df[fold != k], classes)
        routes[f"chosen_on_fold{1 - k}"] = sorted(route)
        sub = df[fold == k]
        p, ch = apply(sub, route)
        for idx, pp, cc in zip(sub.index, p, ch):
            preds[idx], changed[idx] = pp, cc
    route_all = sorted(choose_route(df, classes))

    res = {"question_classifier_acc": {"all": qacc, "complexity1": qacc1}, "routes": {**routes, "chosen_on_all": route_all},
           "rows_changed": int(sum(changed)), "beams": args.beams}
    name_c = "consistency_fixed" if args.route else "consistency_crossfit"
    for name, p in [("greedy", df.greedy.tolist()), (name_c, preds)]:
        m = per_sample(p, df.answer.tolist())
        d = df.assign(**{c: m[c].values for c in TEXT_METRICS}, prediction=p)
        s = summarize(d, p, df.answer.tolist())
        ft = fact_table(d[d.complexity == 1])
        fs = fact_summary(ft)
        ckr = ft[ft.checkable].assign(correct=lambda x: x.correct.astype(float))
        fs["overall"]["ci95"] = bootstrap_ci(ckr, ["correct"])["correct"]
        res[name] = {"overall": s["overall"], "rougeL_ci95": bootstrap_ci(d, ["rougeL"])["rougeL"], "facts_complexity1": fs}
        if name != "greedy":
            df[f"pred_{name}"] = p
    # paired image-bootstrap CI of the fact-accuracy gain
    ft0 = fact_table(df.assign(prediction=df.greedy)[c1]); ft1 = fact_table(df.assign(prediction=preds)[c1])
    ck = ft0.checkable.values
    diff = pd.DataFrame({"img_id": ft0.img_id[ck].values,
                         "d": ft1.correct[ck].values.astype(float) - ft0.correct[ck].values.astype(float)})
    res["fact_gain_ci95"] = bootstrap_ci(diff, ["d"])["d"]
    res["fact_gain"] = float(diff.d.mean())
    tag = f"{args.split}" + ("_fixed" if args.route else "") + (f"_limit{args.limit}" if args.limit else "")
    df.assign(changed=changed)[["orig_idx", "img_id", "complexity", "question", "answer", "greedy",
                                f"pred_{name_c}", "changed"]].to_json(
        os.path.join(out_dir, f"{tag}_predictions_consistency.jsonl"), orient="records", lines=True)
    json.dump(res, open(os.path.join(out_dir, f"{tag}_consistency_metrics.json"), "w"), indent=1, default=float)
    g, c = res["greedy"], res[name_c]
    print(f"question classifier acc {qacc:.4f} (complexity-1 {qacc1:.4f}) | routes {res['routes']} | rows changed {res['rows_changed']}")
    for n, r in [("greedy", g), ("consistency", c)]:
        o, f = r["overall"], r["facts_complexity1"]["overall"]
        print(f"{n:12s} ROUGE-L {o['rougeL']:.4f} METEOR {o['meteor']:.4f} EM {o['exact_match']:.4f} | fact acc "
              f"{f['fact_acc']:.4f} {[round(x, 4) for x in f['ci95']]} macro {f['macro_fact_acc']:.4f}")
    print(f"fact gain {res['fact_gain']:+.4f} CI {[round(x, 4) for x in res['fact_gain_ci95']]}")
    bc0, bc1 = g["facts_complexity1"]["by_class"], c["facts_complexity1"]["by_class"]
    for k in bc0:
        if abs(bc0[k]["fact_acc"] - bc1[k]["fact_acc"]) > 1e-9:
            print(f"   {k:22s} {bc0[k]['fact_acc']:.3f} -> {bc1[k]['fact_acc']:.3f} (n={int(bc0[k]['n'])})")


if __name__ == "__main__":
    main()
