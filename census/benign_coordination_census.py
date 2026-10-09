#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------
# Paper: "The Limits of Query Monitoring for Distributed Model Extraction"
# Implements: the benign-coordination census recommended in §9 "Guidance for providers", item 1
#   ("Measure your benign coordination first"), as a standalone tool for a provider's own logs.
# Quantities and where the paper reports them for WildChat:
#   ratio_lambda  real/shuffled top eigenvalue of R_c  -> §5 'Padding, and a fix that looks at directions' (1.14,
#                 development weeks; computed in experiments/02_development_weeks/step23_hollowed_calibration_scale.py)
#                 and §7 'What did not' (2.22 MiniLM / 3.06 E5, untouched weeks; computed in
#                 experiments/04_frozen_holdout/step35_frozen_holdout.py)
#   k_outliers    outlier eigenvectors above the 1.05 x median-matched bulk edge, participation >= 5
#                 -> the multi-outlier rule of §3 (experiments/02_development_weeks/step27_31_*.py)
#   q_task        share of majority-on-task accounts -> Corollary 1 (q = 0.25%, computed in
#                 experiments/03_capability_frontier/step32_written_attacker.py)
# NOTE: this script re-implements the measurement compactly (PCA to 32 dimensions, one-window lagged
# reference). The paper's values were computed inside the Step 23 / Step 35 pipelines (full 384-d
# embeddings, four-week lagged reference), so on WildChat expect the same qualitative picture, not
# identical digits. To reproduce the paper's exact numbers, run those scripts.
# ---------------------------------------------------------------------------------------------
"""Benign-coordination census: measure, in your own traffic, the quantities that decide what a
query-only extraction monitor can see (see "The Limits of Query Monitoring for Distributed
Model Extraction", Guidance for providers, item 1).

Input: an .npz file with
    emb      (N, d) float  query embeddings (any sentence encoder)
    account  (N,)   int    account (or linked-account cluster) id per query
    window   (N,)   int    monitoring window per query, e.g. ISO week
    task     (N,)   float  optional: score in [0, 1] that the query is in a valuable task region

For every window (from the second on, so a lagged reference exists) it reports:
    ratio_lambda      real / shuffled top eigenvalue of the centred, hollowed (correlation) matrix
                      of whitened account rows -- coordination net of individual topicality and drift
    k_outliers        eigenvectors above a median-matched Marchenko-Pastur bulk edge with
                      participation number >= 5 -- the number of benign coordinated groups per window
    topicality        quantiles of per-account log S (how topical individual users are)
    q_task            share of accounts with a majority of on-task queries (Corollary 1), if task is given
Usage: python benign_coordination_census.py traffic.npz --min-queries 10 --modes 300 --out census.json
"""
import argparse, json
import numpy as np


def pca_scale(emb, d_keep=32, total_var=4.0, seed=0):
    x = emb - emb.mean(0)
    sub = x[np.random.default_rng(seed).choice(len(x), min(len(x), 200_000), replace=False)]
    ev, V = np.linalg.eigh(np.cov(sub, rowvar=False)); V = V[:, ::-1][:, :d_keep]; ev = ev[::-1][:d_keep]
    return (x @ V) / np.sqrt(ev.sum() / total_var)


def bank(ref, h, modes, seed):
    rng = np.random.default_rng(seed); W = rng.standard_normal((ref.shape[1], modes)) / h
    F = np.hstack([np.cos(ref @ W), np.sin(ref @ W)]); mu = F.mean(0); C = np.cov(F, rowvar=False) + 1e-6 * np.eye(F.shape[1])
    ev, U = np.linalg.eigh(C); return dict(W=W, mu=mu, T=U / np.sqrt(np.maximum(ev, 1e-9)))


def rows(x, lab, b):
    F = np.hstack([np.cos(x @ b["W"]), np.sin(x @ b["W"])]) - b["mu"]; A = lab.max() + 1
    S = np.zeros((A, F.shape[1])); np.add.at(S, lab, F); n = np.bincount(lab, minlength=A)
    return (S / np.sqrt(np.maximum(n, 1))[:, None]) @ b["T"], n


def mp_median(c):
    a, b = (1 - np.sqrt(c)) ** 2, (1 + np.sqrt(c)) ** 2; x = np.linspace(a + 1e-9, b - 1e-9, 20001)
    f = np.sqrt(np.maximum((b - x) * (x - a), 0)) / (2 * np.pi * c * x); F = np.cumsum(f); F /= F[-1]
    return float(x[np.searchsorted(F, 0.5)])


def spectrum(Y, kmax=20):
    Yc = Y - Y.mean(0, keepdims=True); Yn = Yc / np.maximum(np.linalg.norm(Yc, axis=1, keepdims=True), 1e-12)
    A, r = Yn.shape
    if A >= r:
        ev, U = np.linalg.eigh(Yn.T @ Yn); o = np.argsort(-ev); ev, U = ev[o], U[:, o]
        V = Yn @ U[:, :kmax] / np.sqrt(np.maximum(ev[:kmax], 1e-12))
    else:
        ev, V = np.linalg.eigh(Yn @ Yn.T); o = np.argsort(-ev); ev, V = ev[o], V[:, o][:, :kmax]
    g = A / r; nz = ev[: min(A, r)]
    edge = np.median(nz) / (max(g, 1.0) * mp_median(min(g, 1 / g))) * (1 + np.sqrt(g)) ** 2 - 1
    k = sum(1 for j in range(V.shape[1]) if ev[j] - 1 > 1.05 * edge and 1.0 / np.sum(V[:, j] ** 4) >= 5)
    return float(ev[0] - 1), int(k), float(edge)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--min-queries", type=int, default=10)
    ap.add_argument("--modes", type=int, default=300); ap.add_argument("--h", type=float, default=4.0)
    ap.add_argument("--max-ref", type=int, default=100_000); ap.add_argument("--kmax", type=int, default=50); ap.add_argument("--out", default="census.json"); a = ap.parse_args()
    D = np.load(a.npz); X = pca_scale(D["emb"].astype(np.float64)); acc, win = D["account"], D["window"]; task = D["task"] if "task" in D else None
    out, windows = [], np.unique(win); rng = np.random.default_rng(1)
    for i, w in enumerate(windows[1:], 1):
        ref_idx = np.flatnonzero(win == windows[i - 1]); ref_idx = rng.choice(ref_idx, min(len(ref_idx), a.max_ref), replace=False)
        b = bank(X[ref_idx], a.h, a.modes, seed=7)
        idx = np.flatnonzero(win == w); u, lab, cnt = np.unique(acc[idx], return_inverse=True, return_counts=True)
        keep = cnt[lab] >= a.min_queries; idx, lab = idx[keep], np.unique(lab[keep], return_inverse=True)[1]
        if lab.size == 0 or lab.max() + 1 < 20: continue
        Y, n = rows(X[idx], lab, b); lam, k, edge = spectrum(Y, a.kmax)
        Ys, _ = rows(X[idx[rng.permutation(len(idx))]], lab, b); lam_s, k_s, _ = spectrum(Ys, a.kmax)
        logS = np.log(np.maximum((Y ** 2).mean(1), 1e-12))
        rec = dict(window=int(w), kmax=a.kmax, k_at_cap=bool(k >= a.kmax), accounts=int(lab.max() + 1), queries=int(len(idx)), lambda_real=lam, lambda_shuffled=lam_s,
                   ratio_lambda=lam / max(lam_s, 1e-12), k_outliers=k, k_outliers_shuffled=k_s, bulk_edge=edge,
                   topicality_q10_q50_q90=[float(v) for v in np.quantile(logS, [0.1, 0.5, 0.9])])
        if task is not None:
            share = np.bincount(lab, weights=(task[idx] >= 0.5).astype(float)) / np.bincount(lab); rec["q_task"] = float(np.mean(share >= 0.5))
        out.append(rec); print(json.dumps(rec))
    summ = {k_: float(np.median([r[k_] for r in out])) for k_ in ["ratio_lambda", "k_outliers", "k_outliers_shuffled"]} if out else {}
    if out and "q_task" in out[0]: summ["q_task"] = float(np.median([r["q_task"] for r in out]))
    json.dump(dict(windows=out, median=summ), open(a.out, "w"), indent=1); print("median over windows:", json.dumps(summ))


if __name__ == "__main__":
    main()
