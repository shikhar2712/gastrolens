"""End-to-end pipeline: prepare -> train -> evaluate -> consistency -> [judge] -> export.

  python -m kvqa.pipeline --config experiments/kvasir_vqa_x1_final/config.yaml
  python -m kvqa.pipeline --config ... --judge-limit 20        # also grade 20 answers per split with the local LLM judge
  python -m kvqa.pipeline --config ... --judge-limit 0          # grade every answer (hours of GPU time)
  python -m kvqa.pipeline --config ... --sanity                 # tiny end-to-end smoke test (writes to <exp>/sanity)
  python -m kvqa.pipeline --config ... --stages evaluate,export --force   # re-run selected stages

Every stage is skipped when its outputs already exist and are newer than the checkpoint (so an interrupted run resumes
and a retrain invalidates downstream results), unless --force. --force never rewrites the shared data cache unless
"prepare" is named explicitly in --stages (and never in --sanity mode).
Stages run as subprocesses of the existing entry points, with logs in <exp>/logs/pipeline_<stage>_<split>.log.
"""
import argparse, glob, json, os, subprocess, sys, time

import pandas as pd
import yaml

from .consistency import DEFAULT_ROUTE

STAGES = ["prepare", "train", "evaluate", "consistency", "judge", "export"]
CACHE_FILES = ["splits.parquet", "attr_schema.json", "image_attrs.json", "answer_tokenizer.json", "image_ids.json"]


class Runner:
    def __init__(self, exp, force):
        self.exp, self.force = exp, force
        self.log = open(os.path.join(exp, "logs", "pipeline.log"), "a")
        self.say(f"=== pipeline start {time.strftime('%Y-%m-%d %H:%M:%S')} | argv {' '.join(sys.argv)}")

    def say(self, msg):
        print(msg, flush=True)
        self.log.write(msg + "\n"); self.log.flush()

    def run(self, name, cmd, fresh_log=False):
        log_path = os.path.join(self.exp, "logs", f"pipeline_{name}.log")
        self.say(f"[{name}] $ {' '.join(cmd)}  (log: {log_path})")
        t0 = time.time()
        env = {**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"), "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w" if fresh_log else "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        self.say(f"[{name}] exit {rc} in {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            tail = open(log_path).read().splitlines()[-15:]
            self.say("\n".join(tail))
            raise SystemExit(f"stage {name} failed (exit {rc}); see {log_path}")

    def skip(self, name, *paths, newer_than=None):
        """Skip when every output exists and (if given) is newer than `newer_than` (e.g. the checkpoint)."""
        if self.force or not paths or not all(os.path.exists(p) for p in paths):
            return False
        if newer_than and os.path.exists(newer_than) and any(os.path.getmtime(p) < os.path.getmtime(newer_than) for p in paths):
            self.say(f"[{name}] outputs are older than {os.path.relpath(newer_than, self.exp)}: re-running")
            return False
        self.say(f"[{name}] skip: outputs exist ({', '.join(os.path.relpath(p, self.exp) for p in paths)})")
        return True


def training_done(exp):
    """best.pt exists and training completed: DONE.json marker (written by kvqa.train at the end, removed at the start
    of a new run), or, for runs started before the marker existed, a 'done in' line in a log newer than best.pt."""
    ck = os.path.join(exp, "checkpoints", "best.pt")
    if not os.path.exists(ck):
        return False
    if os.path.exists(os.path.join(exp, "checkpoints", "DONE.json")):
        return True
    logs = [os.path.join(exp, "logs", f) for f in ("train_stdout.log", "pipeline_train.log")]
    return any(os.path.exists(l) and os.path.getmtime(l) >= os.path.getmtime(ck) and "done in" in open(l).read() for l in logs)


def training_running(config, sanity=False):
    """PIDs of live `kvqa.train` processes writing to the same place (same config and same --sanity mode), so the
    pipeline never starts a second, clashing run. Sanity runs write to <exp>/sanity and do not clash with full runs."""
    target = os.path.abspath(config)
    pids = []
    for d in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            args = open(d, "rb").read().split(b"\0")
        except OSError:
            continue
        args = [x.decode(errors="ignore") for x in args if x]
        if "kvqa.train" in args and "--config" in args:
            cfg = args[args.index("--config") + 1] if args.index("--config") + 1 < len(args) else ""
            cwd = os.readlink(os.path.dirname(d) + "/cwd") if os.path.exists(os.path.dirname(d) + "/cwd") else ""
            same_mode = ("--sanity" in args) == sanity
            if same_mode and os.path.abspath(os.path.join(cwd, cfg)) == target and int(d.split("/")[2]) != os.getpid():
                pids.append(int(d.split("/")[2]))
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default=None, help=f"comma-separated subset of {','.join(STAGES)} (default: all)")
    ap.add_argument("--force", action="store_true", help="re-run the selected stages even if outputs exist")
    ap.add_argument("--sanity", action="store_true", help="tiny smoke test: short training, small eval subsets")
    ap.add_argument("--judge-limit", type=int, default=None,
                    help="grade N answers per split with the local LLM judge (0 = all; omit = skip judge)")
    a = ap.parse_args()
    explicit = a.stages is not None
    asked = a.stages.split(",") if explicit else STAGES
    unknown = [s for s in asked if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {STAGES}")
    stages = [s for s in STAGES if s in asked]
    cfg = yaml.safe_load(open(a.config))
    root = os.path.dirname(os.path.abspath(a.config))
    exp = os.path.join(root, "sanity") if a.sanity else root
    os.makedirs(os.path.join(exp, "logs"), exist_ok=True)
    R = Runner(exp, a.force)
    py = sys.executable
    cache = cfg["data"]["cache_dir"]
    lim = ["--limit", "3"] if a.sanity else []
    ckpt = os.path.join(exp, "checkpoints", "best.pt")

    if "prepare" in stages:
        files = [os.path.join(cache, f) for f in CACHE_FILES] + [os.path.join(cache, f"images_{cfg['data']['image_size']}.npy")]
        rebuild = a.force and explicit and not a.sanity  # never rewrite the shared cache implicitly
        if all(os.path.exists(f) for f in files) and not rebuild:
            R.say(f"[prepare] skip: data cache {cache} complete")
        else:
            R.run("prepare", [py, "-m", "kvqa.prepare", "--config", a.config])

    trained_now = False
    if "train" in stages and a.force and not a.sanity and os.path.exists(os.path.join(root, "RELEASE")):
        raise SystemExit(f"[train] {root} is marked RELEASE: --force would overwrite the released checkpoint. To reproduce it, "
                         "copy its config.yaml into a new experiment folder and run the pipeline there (make reproduce).")
    if "train" in stages:
        running = training_running(a.config, a.sanity)
        if running:
            raise SystemExit(f"[train] kvqa.train is already running for this config (pid {running}); "
                             f"re-run the pipeline when it finishes (it will then skip training)")
        if not a.force and training_done(exp):
            R.say("[train] skip: finished checkpoint exists")
        else:
            R.run("train", [py, "-m", "kvqa.train", "--config", a.config] + (["--sanity"] if a.sanity else []), fresh_log=True)
            trained_now = True
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt}; run the train stage first")

    splits = [s for s in ["test", "heldout"] if s in set(pd.read_parquet(f"{cache}/splits.parquet", columns=["split"]).split)]
    tag = "_limit3" if a.sanity else ""
    for s in splits:
        if "evaluate" in stages and not R.skip(f"evaluate:{s}", os.path.join(exp, "metrics", f"{s}{tag}_metrics.json"),
                                               newer_than=ckpt):
            R.run(f"evaluate_{s}", [py, "-m", "kvqa.evaluate", "--exp", exp, "--split", s] + lim)
        cons = os.path.join(exp, "consistency", f"{s}_fixed{tag}_predictions_consistency.jsonl")
        cons_m = os.path.join(exp, "consistency", f"{s}_fixed{tag}_consistency_metrics.json")
        if "consistency" in stages and not R.skip(f"consistency:{s}", cons, cons_m, newer_than=ckpt):
            R.run(f"consistency_{s}", [py, "-m", "kvqa.consistency", "--exp", exp, "--split", s,
                                       "--route", ",".join(DEFAULT_ROUTE)] + lim + (["--regen"] if a.force or trained_now else []))
        if "judge" in stages and a.judge_limit is not None:
            out = os.path.join(exp, "judge", f"{s}_judge{tag}" + (f"_n{a.judge_limit}" if a.judge_limit else "_all") + ".jsonl")
            if a.force:
                for p in (out, os.path.splitext(out)[0] + "_summary.json"):
                    if os.path.exists(p):
                        os.remove(p)
            # kvqa.judge itself re-grades any row whose answer changed since it was last graded
            R.run(f"judge_{s}", [py, "-m", "kvqa.judge", "run", "--pred", cons, "--pred-col", "pred_consistency_fixed",
                                 "--cache", cache, "--split", s, "--out", out]
                  + (["--limit", str(a.judge_limit)] if a.judge_limit else []))
            if s == "test" and not a.judge_limit and not a.sanity:  # head-to-head with the paper's graded predictions
                R.run("judge_compare", [py, "-m", "kvqa.judge", "compare", "--eval", out,
                                        "--out", os.path.join(exp, "judge", "comparison_vs_published.json")])

    if "export" in stages:
        export(exp, cfg, splits, tag, R)
    R.say(f"=== pipeline done {time.strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------- export: final answers + model card ----------------
def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def export(exp, cfg, splits, tag, R):
    import torch
    os.makedirs(os.path.join(exp, "final"), exist_ok=True)
    ck = os.path.join(exp, "checkpoints", "best.pt")
    ckd = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = ckd["config"]  # the configuration the checkpoint was actually trained with (e.g. sanity overrides)
    t = cfg["train"]
    sanity_note = (f" SANITY RUN: at most {t.get('max_steps')} steps on {t.get('train_subset_images')} images; numbers are "
                   f"meaningless." if t.get("train_subset_images") else "")
    lines = [f"# Model card: {cfg['experiment']}", "",
             f"Generated by `kvqa.pipeline` on {time.strftime('%Y-%m-%d %H:%M')}. Checkpoint: `checkpoints/best.pt` "
             f"({os.path.getsize(ck) / 1e9:.2f} GB). Config: `config.yaml`. Inference: `python -m kvqa.predict --exp {os.path.relpath(exp)} "
             f"--image <img.jpg> --question \"...\"`.", "",
             f"Recipe: {cfg['model']['vision']} image encoder (last {cfg['model']['vision_trainable_blocks']} blocks trained) + "
             f"{cfg['model']['text']} question encoder, custom fusion layers, fact-classifier heads, and a "
             f"{cfg['model']['dec_layers']}-layer answer decoder trained from scratch. {t['epochs']} epochs, bf16, seed "
             f"{t.get('seed', cfg['seed'])}; best checkpoint = epoch {ckd['epoch']} (step {ckd['step']}).{sanity_note} Answers use greedy decoding plus the answer-consistency step for "
             f"{', '.join(DEFAULT_ROUTE)} (sentence re-chosen among 8 beams to agree with the fact classifier).", ""]
    for s in splits:
        m = _load(os.path.join(exp, "metrics", f"{s}{tag}_metrics.json"))
        c = _load(os.path.join(exp, "consistency", f"{s}_fixed{tag}_consistency_metrics.json"))
        stale = os.path.join(exp, "final", f"{s}_answers.jsonl")
        if not c and os.path.exists(stale):
            os.remove(stale)  # never leave answers from an older run next to new metrics
        if not m:
            R.say(f"[export] WARNING: no metrics for split {s}; run the evaluate stage")
            continue
        if not c:
            R.say(f"[export] WARNING: no consistency results for split {s}; final answers not exported")
        o = m["overall"]
        f = m.get("facts_complexity1", {}).get("overall", {})
        lines += [f"## {s} split ({int(o['n'])} questions)", "",
                  "| decoding | ROUGE-L | METEOR | BLEU | exact match | clinical fact accuracy (complexity 1) |",
                  "|---|---|---|---|---|---|",
                  f"| greedy | {o['rougeL']:.4f} | {o['meteor']:.4f} | {o['bleu_corpus']:.4f} | {o['exact_match']:.4f} | "
                  f"{f.get('fact_acc', float('nan')):.4f} |"]
        if c:
            g0 = c["greedy"]["facts_complexity1"]["overall"]["fact_acc"]
            co, cf = c["consistency_fixed"]["overall"], c["consistency_fixed"]["facts_complexity1"]["overall"]
            lines.append(f"| greedy + consistency (final) | {co['rougeL']:.4f} | {co['meteor']:.4f} | {co['bleu_corpus']:.4f} | "
                         f"{co['exact_match']:.4f} | {cf['fact_acc']:.4f} (gain {c['fact_gain'] * 100:+.2f} pts, "
                         f"95% CI [{c['fact_gain_ci95'][0] * 100:+.2f}, {c['fact_gain_ci95'][1] * 100:+.2f}], paired against "
                         f"its own greedy re-run {g0:.4f}) |")
            src = os.path.join(exp, "consistency", f"{s}_fixed{tag}_predictions_consistency.jsonl")
            df = pd.read_json(src, lines=True).rename(columns={"answer": "reference_answer", "greedy": "greedy_answer",
                                                               "pred_consistency_fixed": "answer"})
            df.to_json(os.path.join(exp, "final", f"{s}_answers.jsonl"), orient="records", lines=True)
        th = m.get("text_prior_baseline", {}).get("overall")
        if th:
            lines.append(f"| no-image baseline | {th['rougeL']:.4f} | {th['meteor']:.4f} | {th['bleu_corpus']:.4f} | "
                         f"{th['exact_match']:.4f} | {m['text_prior_baseline'].get('facts_complexity1', {}).get('overall', {}).get('fact_acc', float('nan')):.4f} |")
        jfs = sorted(glob.glob(os.path.join(exp, "judge", f"{s}_judge{tag}_*_summary.json")))
        if any(f.endswith("_all_summary.json") for f in jfs):  # a full grading supersedes samples
            jfs = [f for f in jfs if f.endswith("_all_summary.json")]
        for jf in jfs:
            j = _load(jf)["ours"]
            if j.get("per_fact_score") is None:
                lines += ["", f"LLM judge ({os.path.basename(jf)}): no parseable grades (unparsed {j['rows_unparsed']})."]
                continue
            lines += ["", f"LLM judge ({os.path.basename(jf)}; Qwen3-30B-A3B, paper protocol): per-fact score "
                          f"{j['per_fact_score']:.4f} (mean over fact types {j['per_class_mean']:.4f}), rows fully correct "
                          f"{j['rows_all_correct']:.4f}, rows graded {j['rows_graded']} (unparsed {j['rows_unparsed']})."]
        comps = [(f, _load(os.path.join(exp, "judge", f))) for f in ("comparison_vs_published_samejudge.json", "comparison_vs_published.json")]
        comps = [(f, c) for f, c in comps if c] if s == "test" else []
        for comp_file, comp in comps:
            grader = ("all answers graded by our local copy of the judge" if "samejudge" in comp_file
                      else "published models graded by the paper's own judge runs")
            lines += ["", f"({grader}; `judge/{comp_file}`)"]
            lines += ["Head-to-head with the paper's published fine-tuned models, same questions, same judge protocol "
                      "(paired; 95% CIs resample images):", "",
                      "| model | facts correct | gain vs model [95% CI] | answers fully correct gain [95% CI] |", "|---|---|---|---|"]
            for name, v in comp["vs"].items():
                lines.append(f"| {name} | {v['theirs_per_fact'] * 100:.2f}% (ours {v['ours_per_fact'] * 100:.2f}%) | "
                             f"{v['gain_per_fact'] * 100:+.2f} [{v['gain_per_fact_ci95'][0] * 100:+.2f}, {v['gain_per_fact_ci95'][1] * 100:+.2f}] | "
                             f"{v['gain_rows_all_correct'] * 100:+.2f} [{v['gain_rows_all_correct_ci95'][0] * 100:+.2f}, "
                             f"{v['gain_rows_all_correct_ci95'][1] * 100:+.2f}] |")
        lines.append("")
    if "heldout" not in splits:
        lines += ["## Generalisation to new images", "",
                  "This model was trained on all official training images, so it has no held-out images of its own. Its expected "
                  "accuracy on never-seen images is taken from the same recipe trained with 620 images held out "
                  "(kvasir_vqa_x1_imgdisjoint and its seed repeats; experiments/analyses/seed_noise_2026-09-28.md): ROUGE-L 0.7205 ± 0.0012, "
                  "clinical fact accuracy 90.3% ± 0.3 (greedy), about 90.7% with the consistency step.", ""]
    lines += ["## Caveats", "",
              "- The official test reuses 94% of training images, so its scores overstate performance on new images.",
              "- Clinical fact accuracy is an automated rule-based check against the dataset's labels (which are noisy for polyp "
              "size, type and colour); it covers ~80% of single-fact questions and has not been verified by a clinician.",
              "- Research prototype under a non-commercial dataset licence (CC BY-NC 4.0); not a medical device.", ""]
    open(os.path.join(exp, "MODEL_CARD.md"), "w").write("\n".join(lines))
    R.say(f"[export] wrote {os.path.join(exp, 'MODEL_CARD.md')} and final/*_answers.jsonl")


if __name__ == "__main__":
    main()
