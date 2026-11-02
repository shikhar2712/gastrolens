"""Re-score the official fine-tuned baselines' published test predictions with OUR metric code.

Source: github.com/simula/Kvasir-VQA-x1 results/eval/1_*_eval.json (Track 1, original images).
python -m kvqa.score_official --exp experiments/kvasir_vqa_x1_baseline
"""
import argparse, json, os

import pandas as pd

from .metrics import TEXT_METRICS, per_sample, summarize

FILES = {
    "Qwen2.5-VL-7B-ft (official, v0-3333)": "data/official_preds/1_transform_to_vqa_format_test_pred_v0-3333_eval.json",
    "MedGemma-4B-ft (official, 3952)": "data/official_preds/1_transform_to_vqa_format_test_pred_medgemma-3952_eval.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args()
    df = pd.read_parquet(f"{args.cache}/splits.parquet")
    test = df[df.split == "test"].reset_index(drop=True)
    out = {}
    for name, path in FILES.items():
        off = pd.DataFrame(json.load(open(path)))
        n_raw = len(off)
        off = off.drop_duplicates(["img_id", "question", "answer"])
        m = test.merge(off[["img_id", "question", "answer", "response"]], on=["img_id", "question", "answer"], how="left")
        missing = int(m.response.isna().sum())
        m = m[m.response.notna()].reset_index(drop=True)
        preds = m.response.astype(str).str.strip().tolist()
        s = per_sample(preds, m.answer.tolist())
        for c in TEXT_METRICS:
            m[c] = s[c].values
        res = summarize(m, preds, m.answer.tolist())
        res["alignment"] = {"official_rows": n_raw, "official_unique": len(off), "our_test_rows": len(test),
                            "matched": len(m), "unmatched_test_rows": missing, "source": path}
        out[name] = res
        o = res["overall"]
        print(f"{name}: matched {len(m)}/{len(test)} (official rows {n_raw}) | EM {o['exact_match']:.4f} "
              f"BLEU {o['bleu_corpus']:.4f} R1 {o['rouge1']:.4f} R2 {o['rouge2']:.4f} RL {o['rougeL']:.4f} METEOR {o['meteor']:.4f}")
    os.makedirs(os.path.join(args.exp, "metrics"), exist_ok=True)
    json.dump(out, open(os.path.join(args.exp, "metrics", "official_baselines_rescored.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
