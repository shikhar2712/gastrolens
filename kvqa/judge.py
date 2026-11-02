"""LLM-as-a-judge, replicating the Kvasir-VQA-x1 paper's adjudicator (Table 7) locally.

Source protocol: github.com/simula/Kvasir-VQA-x1 evaluation/llm_adjudicator.py (cloned 2026-09-29). The paper grades each
prediction per clinical aspect (question_class) with a binary score + reason, using Qwen/Qwen3-30B-A3B in non-thinking
mode via vLLM. Here the same model, system prompt, user prompt, "/no_think" suffix and JSON parsing run locally, so no data
leaves the machine and there is no API cost. Two backends:
  --backend vllm  (default; like the paper): an OpenAI-compatible vLLM server on this machine,
      started with scripts/start_judge_server.sh (or `make judge-server`)
  --backend hf    Hugging Face transformers in-process (no server; much slower for this MoE model).
Difference from the paper: greedy decoding (temperature 0); the paper used the server's default sampling.

  # grade a prediction file (evaluate or consistency output); --limit N grades a fixed random sample of rows
  python -m kvqa.judge run --pred experiments/<exp>/predictions/test_predictions.jsonl --cache data/cache --split test \
      --out experiments/<exp>/judge/test_judge.jsonl [--pred-col prediction] [--limit 20]
  # calibrate: re-grade published Qwen2.5-VL answers and compare with the paper's own grades for the same rows
  python -m kvqa.judge calibrate --published data/official_preds/1_transform_to_vqa_format_test_pred_v0-3333_eval.json \
      --out experiments/analyses/judge_calibration/qwen_v0-3333.jsonl --limit 200
  # summarize a graded file (optionally next to a published *_eval.json restricted to the same rows)
  python -m kvqa.judge summarize --eval <graded.jsonl> [--published <paper *_eval.json>]
"""
import argparse, ast, collections, hashlib, json, os, re, time

import pandas as pd

MODEL = "Qwen/Qwen3-30B-A3B"
# verbatim from llm_adjudicator.py
SYSTEM_PROMPT = (
    "You are an medical examiner evaluating a **doctor's written response** in a "
    "medical exam and always provide a **structured JSON evaluation** of the response."
)


def evaluate_response(question, model_response, complexity, question_class, original, answer):
    """Verbatim copy of the paper's prompt builder (llm_adjudicator.evaluate_response)."""
    aspects_json = ",\n".join(
        f'    "{aspect}": {{\n      "score": 0 or 1,\n      "reason": "<short justification>"\n    }}'
        for aspect in question_class
    )
    return f"""
    ## CONTEXT
    The current **exam question** is derived from one or more original Q/A items (see `original`) and may vary in complexity. It has been annotated with one or more **aspect labels** (see `question_class`), where each label represents a specific area of clinical knowledge.

    ## TASK
    Grade the doctor's response against each individual aspect. For each aspect in `question_class`:
    - Compare the doctor's response to the correct answer, using the original Q/A pairs for context.
    - Assign "score": 1 if the response fully and correctly addresses that aspect, else 0.
    - Give a short reason.

    ## OUTPUT FORMAT (STRICT)
    Return a valid JSON object where each key is one aspect label and each value has "score" and "reason".
    Wrap the JSON in triple backticks with `json`. No extra text.

    ```json
    {{
    {aspects_json}
    }}
    ```

    ## Input:
    Exam Question: {question}
    Doctor's Response: {model_response}
    Correct Answer: {answer}
    Question Complexity Level: {complexity}
    Original Q/A Reference: {original}
    Evaluation Aspects: {question_class}
    """


def parse(text, classes):
    """Same parsing rule as the paper: first ```json block, must contain every aspect."""
    txt = re.findall(r"```json(?:[^\n]*)\n(.*?)```", text, re.DOTALL)[0] if "```" in text else None
    res = json.loads(txt)
    assert all(k in res for k in classes), f"Missing keys: {res}"
    return res


def row_key(img_id, question):
    return hashlib.md5(f"{img_id}\t{question}".encode()).hexdigest()


def sample_keys(keys, limit, seed=0):
    """Deterministic sample: the same (image, question) rows are picked for every model, given the same pool."""
    if not limit or limit >= len(keys):
        return set(keys)
    return set(sorted(keys, key=lambda k: hashlib.md5(f"{seed}:{k}".encode()).hexdigest())[:limit])


# ---------------- local judge ----------------
class LocalJudge:
    def __init__(self, model=MODEL, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()
        self.max_new_tokens = max_new_tokens
        self.name = model

    def chat(self, prompt):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + " /no_think"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(self, prompts):
        enc = self.tok([self.chat(p) for p in prompts], return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


class VllmJudge:
    """Client for an OpenAI-compatible vLLM server; sends the paper's exact messages, concurrently."""

    def __init__(self, url="http://localhost:8000/v1", model=MODEL, max_new_tokens=512, workers=64):
        self.url, self.name, self.max_new_tokens, self.workers = url.rstrip("/"), model, max_new_tokens, workers

    def one(self, prompt):
        import urllib.request
        body = {"model": self.name, "max_tokens": self.max_new_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": [{"type": "text", "text": prompt + " /no_think"}]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - retry transient server errors, then surface the failure as text
                err = e
                time.sleep(2 ** attempt)
        return f"<request failed: {err}>"

    def generate(self, prompts):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self.one, prompts))


def grade(items, out_path, batch=16, model=MODEL, backend="vllm", url="http://localhost:8000/v1"):
    """items: dicts with key, question, response, complexity, question_class (list), original, answer (+ passthrough).
    Appends one JSON line per item to out_path; rows already in out_path are skipped (resumable)."""
    current = {it["key"]: it["response"] for it in items}
    done = set()
    if os.path.exists(out_path):
        old = [json.loads(l) for l in open(out_path)]
        keep = [r for r in old if current.get(r["key"]) == r["response"]]
        if len(keep) != len(old):  # responses changed (e.g. retrained model): drop their stale grades
            print(f"dropping {len(old) - len(keep)} stale graded rows from {out_path}", flush=True)
            with open(out_path, "w") as f:
                for r in keep:
                    f.write(json.dumps(r, default=str) + "\n")
        done = {r["key"] for r in keep}
    todo = [it for it in items if it["key"] not in done]
    print(f"{len(items)} rows, {len(done)} already graded, {len(todo)} to grade", flush=True)
    if not todo:
        return
    judge = VllmJudge(url, model) if backend == "vllm" else LocalJudge(model)
    if backend == "vllm":
        batch = max(batch, 256)  # concurrency comes from the server; big chunks keep it busy
    todo.sort(key=lambda it: len(it["question"]) + len(it["response"]) + len(it["original"]))  # less padding
    t0 = time.time()
    with open(out_path, "a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            prompts = [evaluate_response(it["question"], it["response"], it["complexity"], it["question_class"],
                                         it["original"], it["answer"]) for it in chunk]
            for it, raw in zip(chunk, judge.generate(prompts)):
                rec = {**it, "judge_model": judge.name, "raw": raw}
                try:
                    rec["eval_json"] = parse(raw, it["question_class"])
                except Exception as e:  # noqa: BLE001 - same behaviour as the paper: keep the row, record the error
                    rec["error"] = str(e)
                f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            n = i + len(chunk)
            rate = n / (time.time() - t0)
            print(f"  graded {n}/{len(todo)} | {rate:.2f} rows/s | eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)


# ---------------- inputs ----------------
def load_ours(pred_path, cache, split, pred_col):
    p = pd.read_json(pred_path, lines=True)
    sp = pd.read_parquet(f"{cache}/splits.parquet", columns=["split", "orig_idx", "img_id", "question", "original", "question_class"])
    sp = sp[sp.split == split]
    m = p.drop(columns=[c for c in ["question_class"] if c in p]).merge(sp, on=["orig_idx", "img_id", "question"], how="left")
    assert m.original.notna().all(), "could not match every prediction row to the dataset"
    return [dict(key=row_key(r.img_id, r.question), img_id=r.img_id, orig_idx=int(r.orig_idx), question=r.question,
                 response=str(getattr(r, pred_col)).strip(), answer=r.answer, complexity=str(r.complexity),
                 question_class=[str(c) for c in r.question_class], original=r.original) for r in m.itertuples()]


def load_published(path):
    """The paper's *_eval.json files contain some duplicated (img_id, question) rows from its merge; keep the first."""
    rows, seen = [], set()
    for r in json.load(open(path)):
        k = row_key(r["img_id"], r["question"])
        if k in seen:
            continue
        seen.add(k)
        qc = r["question_class"]
        qc = ast.literal_eval(qc) if isinstance(qc, str) else list(qc)  # stored as "['polyp_type']" in the paper's files
        rows.append(dict(key=row_key(r["img_id"], r["question"]), img_id=r["img_id"], question=r["question"],
                         response=str(r["response"]).strip(), answer=r["answer"], complexity=str(r["complexity"]),
                         question_class=qc, original=r["original"], published_eval=r.get("eval_json")))
    return rows


# ---------------- summaries ----------------
def scores_of(rows, field):
    """Per-row list of (aspect, hit), aggregated like the paper (make_figures.scores_from_eval): rows without an eval
    dict are skipped (counted as missing); an aspect absent from the dict is skipped; hit = (score == 1)."""
    out, missing = [], 0
    for r in rows:
        e = r.get(field)
        if not isinstance(e, dict) or not e:
            missing += 1
            continue
        ss = [(c, int(isinstance(e[c], dict) and e[c].get("score") == 1)) for c in r["question_class"] if c in e]
        if ss:
            out.append((r, ss))
        else:
            missing += 1
    return out, missing


def summarize_scores(scored, missing):
    facts = [s for _, ss in scored for _, s in ss]
    per_cls = collections.defaultdict(list)
    per_cx = collections.defaultdict(list)
    for r, ss in scored:
        for c, s in ss:
            per_cls[c].append(s)
        per_cx[r["complexity"]].append(sum(s for _, s in ss) / len(ss))
    by_class = {k: sum(v) / len(v) for k, v in per_cls.items()}
    return {"rows_graded": len(scored), "rows_unparsed": missing,
            "per_fact_score": sum(facts) / len(facts) if facts else None,
            "per_class_mean": sum(by_class.values()) / len(by_class) if by_class else None,
            "rows_all_correct": sum(all(s for _, s in ss) for _, ss in scored) / len(scored) if scored else None,
            "by_complexity": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_cx.items())},
            "by_class": {k: {"score": round(sum(v) / len(v), 4), "correct": sum(v), "n": len(v)} for k, v in sorted(per_cls.items())}}


def agreement(rows):
    """Aspect-level agreement between our local judge (eval_json) and the paper's grades (published_eval)."""
    a = b = both = 0
    pairs = []
    for r in rows:
        e, p = r.get("eval_json"), r.get("published_eval")
        if not isinstance(e, dict) or not isinstance(p, dict):
            continue
        for c in r["question_class"]:
            if isinstance(e.get(c), dict) and isinstance(p.get(c), dict) and e[c].get("score") in (0, 1) and p[c].get("score") in (0, 1):
                pairs.append((int(e[c]["score"]), int(p[c]["score"])))
    if not pairs:
        return {}
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    px, py = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    pe = px * py + (1 - px) * (1 - py)
    return {"aspects": n, "agreement": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else None,
            "ours_mean": px, "paper_mean": py}


def compare_published(graded_path, published, seed=0, reps=2000, regraded=None):
    """Paired comparison with published models on the same (image, question) rows, both graded per fact.
    published: {name: path to the paper's *_eval.json}. regraded: {name: path to a `calibrate --limit 0` output}, i.e. the
    published answers re-graded by OUR judge, so both sides share one grader; without it the paper's own grades are used.
    95% CIs resample whole images (overall and per fact type)."""
    regraded = regraded or {}
    import numpy as np
    ours = [json.loads(l) for l in open(graded_path)]
    so, miss = scores_of(ours, "eval_json")
    res = {"ours": summarize_scores(so, miss), "vs": {}}
    o = {r["key"]: (r, dict(ss)) for r, ss in so}
    for name, path in published.items():
        if name in regraded:
            pub = [json.loads(l) for l in open(regraded[name])]
            ps, pm = scores_of(pub, "eval_json")  # their answers, our judge
        else:
            pub = load_published(path)
            ps, pm = scores_of([dict(x, eval_json=x["published_eval"]) for x in pub], "eval_json")
        pd_ = {r["key"]: dict(ss) for r, ss in ps}
        rows, per_cls, cls_rows = [], collections.defaultdict(lambda: [0, 0, 0]), []
        for k, (r, a) in o.items():
            b = pd_.get(k)
            common = [c for c in a if b and c in b]
            if not common:
                continue
            rows.append((r["img_id"], sum(a[c] - b[c] for c in common), len(common),
                         sum(a[c] for c in common), sum(b[c] for c in common),
                         float(all(a[c] for c in common)) - float(all(b[c] for c in common))))
            for c in common:
                per_cls[c][0] += a[c]; per_cls[c][1] += b[c]; per_cls[c][2] += 1
                cls_rows.append((c, r["img_id"], a[c] - b[c]))
        df = pd.DataFrame(rows, columns=["img", "dsum", "n", "osum", "psum", "dall"])
        g = df.assign(one=1).groupby("img")[["dsum", "n", "dall", "one"]].sum().to_numpy()
        w = np.random.default_rng(seed).multinomial(len(g), np.full(len(g), 1 / len(g)), size=reps)
        b_fact, b_all = (w @ g[:, 0]) / (w @ g[:, 1]), (w @ g[:, 2]) / (w @ g[:, 3])
        cdf = pd.DataFrame(cls_rows, columns=["cls", "img", "d"])
        cls_ci = {}
        for c, sub in cdf.groupby("cls"):
            gg = sub.groupby("img").d.agg(["sum", "size"]).to_numpy()
            ww = np.random.default_rng(seed).multinomial(len(gg), np.full(len(gg), 1 / len(gg)), size=reps)
            bb = (ww @ gg[:, 0]) / (ww @ gg[:, 1])
            cls_ci[c] = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
        res["vs"][name] = {
            "grader": "our judge (re-graded)" if name in regraded else "paper's own grades",
            "published_all_rows": summarize_scores(ps, pm), "questions": len(df), "our_rows_unmatched": len(o) - len(df),
            "ours_per_fact": df.osum.sum() / df.n.sum(), "theirs_per_fact": df.psum.sum() / df.n.sum(),
            "gain_per_fact": df.dsum.sum() / df.n.sum(), "gain_per_fact_ci95": [float(np.percentile(b_fact, 2.5)), float(np.percentile(b_fact, 97.5))],
            "gain_rows_all_correct": df.dall.mean(), "gain_rows_all_correct_ci95": [float(np.percentile(b_all, 2.5)), float(np.percentile(b_all, 97.5))],
            "per_class": {c: {"ours": v[0] / v[2], "theirs": v[1] / v[2], "n": v[2], "gain_ci95": cls_ci.get(c)}
                          for c, v in sorted(per_cls.items())}}
    return res


PUBLISHED = {"Qwen2.5-VL-7B (fine-tuned, paper)": "data/official_preds/1_transform_to_vqa_format_test_pred_v0-3333_eval.json",
             "MedGemma-4B (fine-tuned, paper)": "data/official_preds/1_transform_to_vqa_format_test_pred_medgemma-3952_eval.json"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("compare", help="paired comparison with the paper's published models (official test only)")
    cp.add_argument("--eval", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--regraded", action="append", default=[],
                    help="NAME=PATH: published answers re-graded by our judge (kvqa.judge calibrate --limit 0)")
    r = sub.add_parser("run")
    r.add_argument("--pred", required=True)
    r.add_argument("--pred-col", default="prediction")
    r.add_argument("--cache", required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    r.add_argument("--url", default="http://localhost:8000/v1")
    c = sub.add_parser("calibrate")
    c.add_argument("--published", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=200)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--model", default=MODEL)
    c.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    c.add_argument("--url", default="http://localhost:8000/v1")
    s = sub.add_parser("summarize")
    s.add_argument("--eval", required=True)
    s.add_argument("--published", default=None)
    a = ap.parse_args()
    if a.cmd == "compare":
        rg = dict(x.split("=", 1) for x in a.regraded)
        res = compare_published(a.eval, {k: v for k, v in PUBLISHED.items() if os.path.exists(v)}, regraded=rg)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        for name, v in res["vs"].items():
            print(f"{name}: ours {v['ours_per_fact'] * 100:.2f}% vs {v['theirs_per_fact'] * 100:.2f}% per fact on {v['questions']} questions; "
                  f"gain {v['gain_per_fact'] * 100:+.2f} pts [{v['gain_per_fact_ci95'][0] * 100:+.2f}, {v['gain_per_fact_ci95'][1] * 100:+.2f}]")
        print("wrote", a.out)
        return
    if a.cmd in ("run", "calibrate"):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        items = load_ours(a.pred, a.cache, a.split, a.pred_col) if a.cmd == "run" else load_published(a.published)
        keep = sample_keys([it["key"] for it in items], a.limit)
        grade([it for it in items if it["key"] in keep], a.out, a.batch, a.model, a.backend, a.url)
        a = argparse.Namespace(eval=a.out, published=getattr(a, "published", None))
    rows = [json.loads(l) for l in open(a.eval)]
    res = {"ours": summarize_scores(*scores_of(rows, "eval_json"))}
    if any("published_eval" in r for r in rows):
        both = [r for r in rows if isinstance(r.get("eval_json"), dict) and isinstance(r.get("published_eval"), dict)]
        res["ours_on_rows_both_graded"] = summarize_scores(*scores_of(both, "eval_json"))
        res["paper_grades_same_rows"] = summarize_scores(*scores_of(both, "published_eval"))
        res["judge_agreement"] = agreement(rows)
    elif a.published:
        pub = {x["key"]: x for x in load_published(a.published)}
        same = [dict(pub[r["key"]], eval_json=pub[r["key"]]["published_eval"]) for r in rows if r["key"] in pub]
        res["published_model_same_rows"] = summarize_scores(*scores_of(same, "eval_json"))
    out = os.path.splitext(a.eval)[0] + "_summary.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk not in ("by_class",)} if isinstance(v, dict) else v
                      for k, v in res.items()}, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
