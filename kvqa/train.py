"""Train the custom Kvasir-VQA-x1 model.

python -m kvqa.train --config experiments/kvasir_vqa_x1_baseline/config.yaml [--sanity]
"""
import argparse, json, math, os, random, shutil, subprocess, sys, time

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from .data import EOS, GroupedVQA
from .facts import fact_summary, fact_table
from .metrics import normalize, per_sample
from .model import KvasirVQAModel


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {c: t.to(dev, non_blocking=True) for c, t in v.items()}
        else:
            out[k] = v
    return out


def decode_ids(atok, ids):
    res = []
    for seq in ids.tolist():
        seq = seq[:seq.index(EOS)] if EOS in seq else seq
        res.append(atok.decode([t for t in seq if t > 3]).strip())
    return res


def save_env(path, cfg):
    env = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "seed": cfg["seed"], "train_seed": cfg["train"].get("seed", cfg["seed"]), "argv": sys.argv}
    env["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split("\n")
    env["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    json.dump(env, open(path, "w"), indent=1)


def loader(ds, cfg, bs):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=ds.collate,
                                       num_workers=cfg["data"]["num_workers"], pin_memory=True,
                                       persistent_workers=False)


@torch.no_grad()
def validate(model, cfg, ds, atok, dev, gen_rows):
    model.eval()
    tot, n, ce_sum = 0, 0, 0.0
    preds, rows = [], []
    for b in loader(ds, cfg, 8):
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, ce, _ = model.losses(b, 0.0, 0.0)
            ce_sum += ce.item() * len(b["rows"]); n += len(b["rows"])
            if len(preds) < gen_rows:
                ids, _ = model.generate(b, cfg["eval"]["max_gen_len"])
                preds += decode_ids(atok, ids)
                rows += b["rows"]
    sub = ds.df.iloc[rows].assign(prediction=preds)
    refs = sub.answer.tolist()
    m = per_sample(preds, refs).mean().to_dict()
    fs = fact_summary(fact_table(sub[sub.complexity == 1]))["overall"]
    model.train()
    return {"val_ce": ce_sum / n, **{f"val_{k}": v for k, v in m.items()}, "val_fact_acc": fs["fact_acc"],
            "val_fact_macro": fs["macro_fact_acc"], "val_gen_n": len(preds)}, list(zip(refs, preds))[:10]


def selection_score(vm, how):
    """Checkpoint-selection score. 'rougeL' = baseline; 'rougeL+facts' also rewards clinically correct answers."""
    return vm["val_rougeL"] if how == "rougeL" else 0.5 * (vm["val_rougeL"] + vm["val_fact_acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    exp_dir = os.path.dirname(os.path.abspath(args.config))
    if args.sanity:
        exp_dir = os.path.join(exp_dir, "sanity")
        cfg["train"].update(epochs=1, max_steps=300, train_subset_images=400, val_gen_rows=300)
    for sub in ["checkpoints", "logs", "model"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    # train.seed varies only training randomness (init, dropout, batch order); top-level seed also fixes the data split
    tseed = cfg["train"].get("seed", cfg["seed"])
    seed_all(tseed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"

    cache = cfg["data"]["cache_dir"]
    atok = Tokenizer.from_file(f"{cache}/answer_tokenizer.json")
    cfg["answer_vocab"] = atok.get_vocab_size()
    train_ds = GroupedVQA(cfg, "train", train=True)
    if cfg["train"]["train_subset_images"]:
        keep = set(sorted(train_ds.df.img_id.unique())[: cfg["train"]["train_subset_images"]])
        train_ds.df = train_ds.df[train_ds.df.img_id.isin(keep)].reset_index(drop=True)
    val_ds = GroupedVQA(cfg, "val", train=False)
    val_ds.groups = val_ds.make_groups(None, k=32)

    model = KvasirVQAModel(cfg, train_ds.attr.schema).to(dev)
    pre, new = model.param_groups()
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in pre + new)
    t = cfg["train"]
    if t.get("vision_lr_decay") or t.get("no_decay_norm_bias"):
        pgroups = model.param_groups(t, t.get("vision_lr_decay"))
    else:  # baseline: two groups, uniform weight decay
        pgroups = [{"params": pre, "lr": t["lr_pretrained"]}, {"params": new, "lr": t["lr_new"]}]
    opt = torch.optim.AdamW(pgroups, weight_decay=t["weight_decay"], betas=(0.9, 0.98), fused=True)
    select = t.get("select_metric", "rougeL")
    B = cfg["data"]["images_per_batch"]
    steps_per_epoch = math.ceil(len(train_ds.make_groups(np.random.default_rng(0))) / B)
    total = t["max_steps"] or steps_per_epoch * t["epochs"]
    warm = max(1, int(total * t["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    # reproducibility artifacts
    yaml.safe_dump(cfg, open(os.path.join(exp_dir, "config_resolved.yaml"), "w"), sort_keys=False)
    save_env(os.path.join(exp_dir, "logs", "env.json"), cfg)
    code_dst = os.path.join(exp_dir, "model", "code")
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)), code_dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["attr_schema.json", "answer_tokenizer.json"]:
        shutil.copy(f"{cache}/{f}", os.path.join(exp_dir, "model", f))
    with open(os.path.join(exp_dir, "model", "architecture.txt"), "w") as f:
        f.write(f"total params {n_total/1e6:.2f}M | trainable {n_train/1e6:.2f}M\n")
        for name, mod in model.named_children():
            f.write(f"{name}: {sum(p.numel() for p in mod.parameters())/1e6:.2f}M total, "
                    f"{sum(p.numel() for p in mod.parameters() if p.requires_grad)/1e6:.2f}M trainable\n")
        f.write("\n" + str(model))
    print(f"params total {n_total/1e6:.1f}M trainable {n_train/1e6:.1f}M | train rows {len(train_ds.df)} "
          f"| steps/epoch {steps_per_epoch} | total steps {total}", flush=True)

    done_marker = os.path.join(exp_dir, "checkpoints", "DONE.json")
    if os.path.exists(done_marker):
        os.remove(done_marker)  # a new run invalidates the previous completion marker
    log = open(os.path.join(exp_dir, "logs", "train.jsonl"), "a")
    step, best, t0 = 0, -1.0, time.time()
    model.train()
    for epoch in range(t["epochs"]):
        train_ds.groups = train_ds.make_groups(np.random.default_rng(tseed + epoch))
        run_ce = run_attr = 0.0
        for b in loader(train_ds, cfg, B):
            b = to_dev(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ce, attr = model.losses(b, t["label_smoothing"], t["attr_loss_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(pre + new, t["grad_clip"])
            opt.step(); sched.step(); step += 1
            run_ce += ce.item(); run_attr += attr.item()
            if step % 50 == 0:
                rec = dict(step=step, epoch=epoch, ce=run_ce / 50, attr=run_attr / 50, grad_norm=gn.item(),
                           lr_new=max(sched.get_last_lr()), elapsed_min=(time.time() - t0) / 60, qa_per_step=len(b["rows"]))
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                run_ce = run_attr = 0.0
            if step >= total:
                break
        vm, samples = validate(model, cfg, val_ds, atok, dev, t["val_gen_rows"])
        rec = dict(step=step, epoch=epoch, elapsed_min=(time.time() - t0) / 60, **vm)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        for r, p in samples[:4]:
            print(f"   GT: {r}\n   PR: {p}", flush=True)
        state = {"model": model.state_dict(), "epoch": epoch, "step": step, "val": vm, "config": cfg}
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
                   os.path.join(exp_dir, "checkpoints", "last.pt"))
        score = selection_score(vm, select)
        if score > best:
            best = score
            torch.save(state, os.path.join(exp_dir, "checkpoints", "best.pt"))
            print(f"   new best val {select} {best:.4f} (epoch {epoch})", flush=True)
        if step >= total:
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump({"step": step, "best_selection_score": best, "minutes": (time.time() - t0) / 60,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(done_marker, "w"))


if __name__ == "__main__":
    main()
