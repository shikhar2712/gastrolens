"""Custom multimodal model for Kvasir-VQA-x1.

Pretrained: SigLIP2 vision tower (last N blocks trainable), BERT question encoder.
Custom:     question-conditioned query fusion, atomic-attribute tokens + heads, answer decoder.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel

# The Space sets this: build the two backbones from their config files only. The checkpoint supplies every weight (load_state_dict is strict, so a
# mismatch fails loudly), and the pretrained weights are not read a second time, which saves about 1 GB of memory at start-up.
LEAN = os.environ.get("GASTROLENS_LEAN") == "1"

from .data import BOS, EOS, PAD


class CrossBlock(nn.Module):
    """Pre-LN: self-attention over x, cross-attention from x to memory, FFN."""

    def __init__(self, d, h, p):
        super().__init__()
        self.n1, self.n2, self.n3, self.nm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mem, x_pad=None):
        h = self.n1(x)
        x = x + self.drop(self.sa(h, h, h, key_padding_mask=x_pad, need_weights=False)[0])
        m = self.nm(mem)
        x = x + self.drop(self.ca(self.n2(x), m, m, need_weights=False)[0])
        return x + self.drop(self.ff(self.n3(x)))


class KvasirVQAModel(nn.Module):
    def __init__(self, cfg, attr_schema):
        super().__init__()
        m = cfg["model"]
        d, h, p = m["d_model"], m["n_heads"], m["dropout"]
        self.vision = SiglipVisionModel(SiglipVisionConfig.from_pretrained(m["vision"])) if LEAN else SiglipVisionModel.from_pretrained(m["vision"])
        self.vision.use_head = False  # unused attention-pooling head; we use patch tokens
        del self.vision.head
        nb = len(self.vision.encoder.layers)
        for n, prm in self.vision.named_parameters():
            prm.requires_grad = n.startswith("post_layernorm") or any(
                n.startswith(f"encoder.layers.{i}.") for i in range(nb - m["vision_trainable_blocks"], nb))
        self.text = (AutoModel.from_config(AutoConfig.from_pretrained(m["text"]), add_pooling_layer=False) if LEAN
                     else AutoModel.from_pretrained(m["text"], add_pooling_layer=False))
        self.text.embeddings.requires_grad_(False)

        self.v_proj = nn.Sequential(nn.Linear(self.vision.config.hidden_size, d), nn.LayerNorm(d))
        self.q_proj = nn.Sequential(nn.Linear(self.text.config.hidden_size, d), nn.LayerNorm(d))
        self.pos2d = nn.Parameter(torch.zeros(1, (cfg["data"]["image_size"] // 16) ** 2, d))

        # --- atomic attribute branch (image-level, question independent) ---
        self.attr_classes = sorted(attr_schema)
        self.attr_schema = attr_schema
        self.attr_queries = nn.Parameter(torch.randn(1, len(self.attr_classes), d) * 0.02)
        self.attr_blocks = nn.ModuleList([CrossBlock(d, h, p) for _ in range(m["attr_layers"])])
        self.attr_norm = nn.LayerNorm(d)
        self.attr_heads = nn.ModuleDict({c: nn.Linear(d, len(attr_schema[c]["labels"])) for c in self.attr_classes})

        # --- question-conditioned fusion ---
        self.queries = nn.Parameter(torch.randn(1, m["n_queries"], d) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(3, d))  # 0 query, 1 question, 2 attribute token
        self.fusion = nn.ModuleList([CrossBlock(d, h, p) for _ in range(m["fusion_layers"])])
        self.fusion_norm = nn.LayerNorm(d)

        # --- answer decoder (from scratch, custom BPE vocab) ---
        dd = m["dec_d_model"]
        self.mem_proj = nn.Linear(d, dd)
        self.vocab = cfg["answer_vocab"]
        self.tok_emb = nn.Embedding(self.vocab, dd, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg["data"]["max_answer_len"], dd)
        layer = nn.TransformerDecoderLayer(dd, m["dec_heads"], 4 * dd, p, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, m["dec_layers"])
        self.dec_norm = nn.LayerNorm(dd)
        self.n_new_init()

    def n_new_init(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def param_groups(self, t=None, vision_lr_decay=None):
        """Without t: (pretrained, new) parameter lists (the baseline's two groups).
        With t (train cfg): AdamW groups with optional layer-wise LR decay over the trainable vision blocks
        (top block at lr_pretrained, each lower block x vision_lr_decay) and, if t['no_decay_norm_bias'],
        no weight decay on biases, norms, embeddings and learned query/position tensors."""
        pre, new = [], []
        for n, prm in self.named_parameters():
            if prm.requires_grad:
                (pre if n.startswith(("vision.", "text.")) else new).append(prm)
        if t is None:
            return pre, new
        nb = len(self.vision.encoder.layers)
        groups = {}
        for n, prm in self.named_parameters():
            if not prm.requires_grad:
                continue
            if n.startswith("vision.encoder.layers."):
                depth = nb - 1 - int(n.split(".")[3])  # 0 = top block
                lr = t["lr_pretrained"] * (vision_lr_decay or 1.0) ** depth
            elif n.startswith(("vision.", "text.")):
                lr = t["lr_pretrained"]
            else:
                lr = t["lr_new"]
            no_wd = t.get("no_decay_norm_bias") and (prm.ndim < 2 or "norm" in n.lower() or "emb" in n
                                                      or n in ("queries", "attr_queries", "pos2d", "type_emb"))
            wd = 0.0 if no_wd else t["weight_decay"]
            groups.setdefault((lr, wd), []).append(prm)
        return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in sorted(groups.items())]

    # ------------------------------------------------------------------
    def encode_images(self, images):
        v = self.vision(pixel_values=images).last_hidden_state
        v = self.v_proj(v) + self.pos2d
        a = self.attr_queries.expand(v.size(0), -1, -1)
        for blk in self.attr_blocks:
            a = blk(a, v)
        a = self.attr_norm(a)
        logits = {c: self.attr_heads[c](a[:, i]) for i, c in enumerate(self.attr_classes)}
        return v, a, logits

    def fuse(self, v, a, img_index, q_ids, q_mask):
        q = self.q_proj(self.text(input_ids=q_ids, attention_mask=q_mask).last_hidden_state)
        B = q.size(0)
        x = torch.cat([self.queries.expand(B, -1, -1) + self.type_emb[0], q + self.type_emb[1],
                       a[img_index] + self.type_emb[2]], 1)
        nq, na = self.queries.size(1), a.size(1)
        pad = torch.cat([q.new_zeros(B, nq, dtype=torch.bool), q_mask == 0, q.new_zeros(B, na, dtype=torch.bool)], 1)
        vm = v[img_index]
        for blk in self.fusion:
            x = blk(x, vm, pad)
        return self.mem_proj(self.fusion_norm(x)), pad

    def decode(self, tgt, mem, mem_pad):
        T = tgt.size(1)
        pos = torch.arange(T, device=tgt.device)
        x = self.tok_emb(tgt) * math.sqrt(self.tok_emb.embedding_dim) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), 1)
        x = self.decoder(x, mem, tgt_mask=causal, tgt_is_causal=True, memory_key_padding_mask=mem_pad)
        return self.dec_norm(x) @ self.tok_emb.weight.t()  # tied output projection

    def forward(self, batch):
        v, a, attr_logits = self.encode_images(batch["images"])
        mem, pad = self.fuse(v, a, batch["img_index"], batch["q_ids"], batch["q_mask"])
        logits = self.decode(batch["ans"][:, :-1], mem, pad)
        return logits, attr_logits

    def losses(self, batch, label_smoothing, attr_w):
        logits, attr_logits = self(batch)
        tgt = batch["ans"][:, 1:]
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                             ignore_index=PAD, label_smoothing=label_smoothing)
        attr_losses = []
        for c, lg in attr_logits.items():
            m = batch["attr_m"][c]
            if m.sum() == 0:
                continue
            t = batch["attr_t"][c]
            if self.attr_schema[c]["type"] == "multi":
                l = F.binary_cross_entropy_with_logits(lg.float(), t, reduction="none").mean(1)
            else:
                l = F.cross_entropy(lg.float(), t.clamp(min=0), reduction="none")
            attr_losses.append((l * m).sum() / m.sum())
        attr = torch.stack(attr_losses).mean() if attr_losses else ce.new_zeros(())
        return ce + attr_w * attr, ce.detach(), attr.detach()

    @torch.no_grad()
    def generate(self, batch, max_len):
        v, a, attr_logits = self.encode_images(batch["images"])
        mem, pad = self.fuse(v, a, batch["img_index"], batch["q_ids"], batch["q_mask"])
        B = mem.size(0)
        seq = torch.full((B, 1), BOS, dtype=torch.long, device=mem.device)
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        for _ in range(max_len - 1):
            nxt = self.decode(seq, mem, pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return seq[:, 1:], attr_logits

    @torch.no_grad()
    def generate_beam(self, batch, max_len, beams=4, length_penalty=1.0, return_all=False):
        """Beam search; hypotheses scored by sum log-prob / length**length_penalty.
        return_all: also return every beam, best first, as (ids (B, K, T), scores (B, K))."""
        v, a, attr_logits = self.encode_images(batch["images"])
        mem, pad = self.fuse(v, a, batch["img_index"], batch["q_ids"], batch["q_mask"])
        B, K, dev = mem.size(0), beams, mem.device
        mem, pad = mem.repeat_interleave(K, 0), pad.repeat_interleave(K, 0)
        seq = torch.full((B * K, 1), BOS, dtype=torch.long, device=dev)
        score = torch.full((B, K), float("-inf"), device=dev)
        score[:, 0] = 0  # all beams start identical; keep one alive
        done = torch.zeros(B * K, dtype=torch.bool, device=dev)
        for _ in range(max_len - 1):
            lp = self.decode(seq, mem, pad)[:, -1].float().log_softmax(-1)  # (B*K, V)
            V = lp.size(-1)
            # finished hypotheses may only extend with PAD at no cost
            lp = torch.where(done[:, None], torch.full_like(lp, float("-inf")), lp)
            lp[done, PAD] = 0
            lp[~done, PAD] = float("-inf")
            lp[:, BOS] = float("-inf")
            cand = (score.reshape(-1, 1) + lp).reshape(B, K * V)
            score, idx = cand.topk(K, -1)
            src = (idx // V) + torch.arange(B, device=dev)[:, None] * K
            tok = (idx % V).reshape(-1)
            src = src.reshape(-1)
            seq = torch.cat([seq[src], tok[:, None]], 1)
            done = done[src] | (tok == EOS)
            if done.all():
                break
        length = ((seq[:, 1:] != PAD) & (seq[:, 1:] != EOS)).sum(1).clamp(min=1).float() + 1  # + EOS
        norm = (score.reshape(-1) / length.pow(length_penalty)).reshape(B, K)
        best = norm.argmax(-1) + torch.arange(B, device=dev) * K
        if return_all:
            order = norm.argsort(-1, descending=True)
            allseq = seq[:, 1:].reshape(B, K, -1).gather(1, order[..., None].expand(-1, -1, seq.size(1) - 1))
            return seq[best, 1:], attr_logits, allseq, norm.gather(1, order)
        return seq[best, 1:], attr_logits
