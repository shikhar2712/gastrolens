"""Group near-duplicate images (e.g. adjacent video frames) so splits can keep each group on one side.

Two images are linked when their 64-bit DCT perceptual hashes differ in <= max_hamming bits AND the Pearson
correlation of their 64x64 RGB thumbnails is >= min_corr; groups are connected components of that graph.
Calibration (2026-09-26): pairs at Hamming 0-2 have median corr 0.99, 3-4: 0.98, 7-8: 0.89, random: 0.42.

python -m kvqa.dedup --cache data/cache --out data/cache/dup_groups.json
"""
import argparse, collections, json

import numpy as np
import torch
from scipy.fft import dctn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="data/cache/dup_groups.json")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--max_hamming", type=int, default=10)
    ap.add_argument("--min_corr", type=float, default=0.90)
    a = ap.parse_args()
    ids = json.load(open(f"{a.cache}/image_ids.json"))
    t = torch.from_numpy(np.array(np.load(f"{a.cache}/images_{a.res}.npy", mmap_mode="r"))).permute(0, 3, 1, 2).float()

    gray = torch.nn.functional.interpolate(t.mean(1, keepdim=True), size=(32, 32), mode="area")[:, 0].numpy()
    d = dctn(gray, axes=(1, 2), norm="ortho")[:, :8, :8].reshape(len(ids), -1)
    bits = torch.from_numpy(d > np.median(d, axis=1, keepdims=True)).float().cuda()
    ham = (bits @ (1 - bits).T + (1 - bits) @ bits.T).cpu().numpy()

    thumb = torch.nn.functional.interpolate(t, size=(64, 64), mode="area").reshape(len(ids), -1)
    thumb = (thumb - thumb.mean(1, keepdim=True)) / thumb.std(1, keepdim=True)
    i, j = np.where(np.triu(ham <= a.max_hamming, 1))
    corr = (thumb[i] * thumb[j]).mean(1).numpy()
    keep = corr >= a.min_corr
    graph = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(len(ids), len(ids)))
    n, lab = connected_components(graph, directed=False)
    json.dump({ids[k]: int(lab[k]) for k in range(len(ids))}, open(a.out, "w"))
    sizes = sorted(collections.Counter(lab).values(), reverse=True)
    print(f"{keep.sum()} links -> {n} groups | largest {sizes[:6]} | images in multi-image groups {sum(s for s in sizes if s > 1)}")


if __name__ == "__main__":
    main()
