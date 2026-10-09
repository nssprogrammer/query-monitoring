# ## Paper numbers reproduced by this script
# Paper: "The Limits of Query Monitoring for Distributed Model Extraction".
# Plain Python script (CPU/numpy, not a Kaggle notebook; see the "Run:" lines in the docstring below for
# runtime). Seeds are fixed (--seed, default 12345);
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# The paper's Step 1 numbers most plausibly come from the Kaggle notebook version of this experiment,
# `step01_missing_mode.py` (same world and statistics, 100 GPU trials, finer padding grid, VERDICT REPORT
# cell). This script computes the same quantities with 30 trials by default; the rows below say where.
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Prediction log (Step 1), row "Union S(k) matches jittered-grid theory" (P1) | relative error of the colluders' union structure factor vs jittered-grid theory | Pass (≤ 2% error) | not isolated (Part A shell table: compare columns S_union and theory) |
# | Prediction log (Step 1), row "Colluder pair correlation −(1 − S̄)/K" (P2) | mean colluder-pair C_ab, measured vs predicted | Pass (−0.0451 vs −0.0451) | Part B, print "colluder-pair mean C_ab ... pred" |
# | Prediction log (Step 1), row "BBP outlier below the bulk" (P3); App. E "Spectral detection condition" ("on a 2-D torus the measured lower outlier matched the BBP prediction to four decimals") | attacked λ_min, measured vs BBP prediction | Pass (0.0702 vs 0.0701) | Part B, prints "lam_min  ATTACK  measured" and "lam_min  ATTACK  BBP prediction" |
# | Prediction log (Step 1), row "Padding threshold f∗ = 0.445" (P4) | predicted padding threshold f* vs empirical f at power 0.5 | 0.445 predicted; Partial (0.408) | Part C, print "predicted threshold f*"; empirical value not isolated (read off the Part C power column; this script's padding grid is coarser and it does not interpolate) |
# | Prediction log (Step 1), row "Outlier independent of K" (P5) | spread (max − min) of attacked λ_min over K ∈ {5, 10, 20, 40, 80} at f = 0.3 | Pass (spread 0.0017) | not isolated (Part D table, lam_min column) |
#
# Not reported in the paper: benign-union S(k), λ_max attack vs null, eigenvector overlap and IPR, single-pair
# SNR, detection power at 1% FPR, per-f and per-K power/overlap tables, step1_figure.png, step1_results.json.
"""
STEP 1 - Toy-world validation of the "missing mode" mechanism
=============================================================
World: unit torus [0,1)^2 (periodic -> no edge effects).
Attacker: space-filling (jittered-grid, hyperuniform) campaign of L*L queries,
          randomly partitioned over K Sybil accounts, optionally padded with a
          fraction f of uniformly random "cover" queries.
Benign:   B accounts, each issuing uniform-random (Poisson-like) queries,
          same per-account size as colluders (fair comparison).
Statistic: per-account Fourier amplitudes rho_a(k)/sqrt(n_a) on m low-k modes,
           cross-account matrix C = F F^* / m  (complex Wishart under the null).

Predictions tested (P1-P5 in the chat):
  P1 union S(k) ~ 1 - sinc^2 sinc^2 ; per-account S ~ 1 - (1-S)/K
  P2 colluder pair C_ab ~ -(1 - S_bar)/K
  P3 outlier BELOW lower MP edge at (1+l)(1+g/l), l = S_bar - 1
  P4 padding threshold f* = 1 - sqrt(g)/(1 - S_bar_design)
  P5 outlier location ~ independent of K

Run:   python step1_missing_mode_toy.py            (full, ~10-20 min)
       python step1_missing_mode_toy.py --quick    (smoke test, <1 min)
Needs: numpy (matplotlib optional, for figures)
"""
import argparse, json, time
import numpy as np

# ----------------------------------------------------------------- geometry
def jittered_grid(L, rng):
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    return (np.stack([i.ravel(), j.ravel()], 1) + rng.random((L * L, 2))) / L

def half_plane_modes(nmax):
    """Integer wavevectors n with 0<|n|<=nmax, one per +/- pair (rho(-k)=conj rho(k))."""
    out = []
    for nx in range(-nmax, nmax + 1):
        for ny in range(0, nmax + 1):
            r2 = nx * nx + ny * ny
            if r2 == 0 or r2 > nmax * nmax or (ny == 0 and nx < 0):
                continue
            out.append((nx, ny))
    return np.array(out, dtype=float)

def fourier(pts, modes, chunk=4096):
    """rho(k) = sum_j exp(-2 pi i n.x_j), chunked to bound memory."""
    acc = np.zeros(len(modes), dtype=complex)
    for s in range(0, len(pts), chunk):
        acc += np.exp(-2j * np.pi * (pts[s:s + chunk] @ modes.T)).sum(0)
    return acc

def S_of(pts, modes):
    return np.abs(fourier(pts, modes)) ** 2 / len(pts)

def S_theory_jitter(modes, L):
    return 1.0 - (np.sinc(modes[:, 0] / L) * np.sinc(modes[:, 1] / L)) ** 2

# ----------------------------------------------------------------- world
def build_world(K, B, L, f_pad, rng, attack=True):
    design = jittered_grid(L, rng) if attack else rng.random((L * L, 2))
    N0 = len(design)
    n_pad = int(round(f_pad / (1.0 - f_pad) * N0)) if f_pad > 0 else 0
    camp = np.vstack([design, rng.random((n_pad, 2))])
    camp = camp[rng.permutation(len(camp))]
    lab = rng.integers(0, K, len(camp))
    colluders = [camp[lab == a] for a in range(K)]
    n_per = len(camp) // K
    benign = [rng.random((n_per, 2)) for _ in range(B)]
    is_col = np.r_[np.ones(K, bool), np.zeros(B, bool)]
    return colluders + benign, is_col, camp

def cross_matrix(accounts, modes):
    F = np.stack([fourier(a, modes) / np.sqrt(len(a)) for a in accounts])
    return F @ F.conj().T / len(modes)

def analyse(accounts, is_col, camp, modes):
    m, A = len(modes), len(accounts)
    g = A / m
    C = cross_matrix(accounts, modes)
    evals, evecs = np.linalg.eigh(C)
    w = np.where(is_col, np.sqrt([len(a) for a in accounts]), 0.0)
    w /= np.linalg.norm(w)
    v = evecs[:, 0]
    S_bar = S_of(camp, modes).mean()
    l = S_bar - 1.0
    lam_pred = (1 + l) * (1 + g / l) if l < -np.sqrt(g) else (1 - np.sqrt(g)) ** 2
    Kc = is_col.sum()
    cc = C[np.ix_(is_col, is_col)].real
    bb = C[np.ix_(~is_col, ~is_col)].real
    off = lambda M: M[~np.eye(len(M), dtype=bool)]
    return dict(
        gamma=g, lam_min=evals[0], lam_2=evals[1], lam_max=evals[-1],
        lo_edge=(1 - np.sqrt(g)) ** 2, hi_edge=(1 + np.sqrt(g)) ** 2,
        lam_pred=lam_pred, S_bar_union=S_bar,
        overlap=np.abs(np.vdot(v, w)) ** 2,
        ipr=np.sum(np.abs(v) ** 4),
        col_pair_mean=off(cc).mean() if Kc > 1 else np.nan,
        ben_pair_mean=off(bb).mean(), ben_pair_sd=off(bb).std(),
    )

def run_trials(T, seed, **kw):
    rows = []
    for t in range(T):
        rng = np.random.default_rng(seed + t)
        acc, is_col, camp = build_world(kw["K"], kw["B"], kw["L"], kw["f"], rng, kw["attack"])
        rows.append(analyse(acc, is_col, camp, kw["modes"]))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

def ms(x):
    return f"{np.nanmean(x):+.4f} ± {np.nanstd(x):.4f}"

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--A", type=int, default=100, help="total accounts K+B")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    T = 5 if a.quick else a.trials
    modes = half_plane_modes(a.nmax)
    m = len(modes)
    B = a.A - a.K
    t0 = time.time()
    print("=" * 72)
    print(f"STEP 1 | L={a.L} (campaign {a.L**2} queries)  nmax={a.nmax}  m={m} modes")
    print(f"       | K={a.K} colluders  B={B} benign  A={a.A}  gamma=A/m={a.A/m:.3f}  trials={T}")
    print("=" * 72)
    res = {"config": vars(a) | {"m": m, "trials": T}}

    # ---------------- Part A: structure factors (P1)
    print("\n[A] Structure factor S(k) by |n| shell  (P1)")
    rng = np.random.default_rng(a.seed)
    rmag = np.linalg.norm(modes, axis=1)
    bins = [1, 4, 8, 12, a.nmax + 1e-9]
    Sth = S_theory_jitter(modes, a.L)
    Su, Sacc, Sben = [], [], []
    for t in range(T):
        acc, is_col, camp = build_world(a.K, B, a.L, 0.0, np.random.default_rng(a.seed + 1000 + t))
        design = camp  # f=0 -> campaign is pure design
        Su.append(S_of(design, modes))
        Sacc.append(np.mean([S_of(x, modes) for x in acc[:a.K]], 0))
        Sben.append(S_of(np.vstack(acc[a.K:a.K + a.K]), modes))  # union of K benign accts
    Su, Sacc, Sben = map(lambda z: np.mean(z, 0), (Su, Sacc, Sben))
    print(f"  {'shell':>10} {'S_union':>9} {'theory':>9} {'S_1acct':>9} {'pred_1acct':>11} {'S_benign_union':>15}")
    rowsA = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (rmag >= lo) & (rmag < hi)
        pred1 = 1 - (1 - Sth[s].mean()) / a.K
        print(f"  [{lo:>2},{hi:>5.0f}) {Su[s].mean():9.4f} {Sth[s].mean():9.4f} "
              f"{Sacc[s].mean():9.4f} {pred1:11.4f} {Sben[s].mean():15.4f}")
        rowsA.append([lo, hi, Su[s].mean(), Sth[s].mean(), Sacc[s].mean(), pred1, Sben[s].mean()])
    res["A"] = rowsA

    # ---------------- Part B: spectrum at f=0 (P2, P3)
    print("\n[B] Spectrum, f=0  (P2, P3)")
    att = run_trials(T, a.seed + 2000, K=a.K, B=B, L=a.L, f=0.0, attack=True, modes=modes)
    nul = run_trials(T, a.seed + 3000, K=a.K, B=B, L=a.L, f=0.0, attack=False, modes=modes)
    thr = np.quantile(nul["lam_min"], 0.01)
    print(f"  MP lower edge (1-sqrt g)^2        : {att['lo_edge'][0]:.4f}")
    print(f"  MP upper edge (1+sqrt g)^2        : {att['hi_edge'][0]:.4f}")
    print(f"  S_bar(union of campaign)          : {ms(att['S_bar_union'])}")
    # [PAPER] Prediction log (Step 1, P3) and App. E: BBP outlier below the bulk, 0.0702 vs 0.0701 (measured vs predicted); plausibly from the notebook version (see header)
    print(f"  lam_min  ATTACK  measured         : {ms(att['lam_min'])}")
    print(f"  lam_min  ATTACK  BBP prediction   : {ms(att['lam_pred'])}")
    print(f"  lam_2    ATTACK (should sit at edge): {ms(att['lam_2'])}")
    print(f"  lam_min  NULL    measured         : {ms(nul['lam_min'])}   1%-quantile={thr:.4f}")
    print(f"  lam_max  ATTACK vs NULL           : {ms(att['lam_max'])}  vs  {ms(nul['lam_max'])}")
    print(f"  |<v_min, w_colluders>|^2          : {ms(att['overlap'])}   (null: {ms(nul['overlap'])})")
    print(f"  IPR(v_min)  (1/K = {1/a.K:.3f})        : {ms(att['ipr'])}")
    # [PAPER] Prediction log (Step 1, P2): colluder-pair correlation, −0.0451 vs −0.0451; plausibly from the notebook version (see header)
    pred_pair = -(1 - att["S_bar_union"].mean()) / a.K
    snr = abs(att["col_pair_mean"].mean()) / att["ben_pair_sd"].mean()
    print(f"  colluder-pair mean C_ab           : {ms(att['col_pair_mean'])}   pred {pred_pair:+.4f}")
    print(f"  benign-pair  mean C_ab            : {ms(att['ben_pair_mean'])}   sd(single pair)={att['ben_pair_sd'].mean():.4f}")
    print(f"  single-pair SNR |signal|/noise    : {snr:.2f}")
    print(f"  detection power @1% FPR (lam_min) : {(att['lam_min'] < thr).mean():.2f}")
    res["B"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in att.items()} | {"null_thr": float(thr)}

    # ---------------- Part C: padding sweep (P4)
    print("\n[C] Padding sweep: attacker adds fraction f of random cover queries  (P4)")
    S_design = att["S_bar_union"].mean()
    g = a.A / m
    # [PAPER] Prediction log (Step 1, P4): predicted padding threshold f* = 0.445 (empirical 0.408 in the notebook version)
    fstar = 1 - np.sqrt(g) / (1 - S_design)
    print(f"  predicted threshold f* = 1 - sqrt(g)/(1-S_design) = {fstar:.3f}")
    print(f"  {'f':>5} {'lam_min':>17} {'lam_pred':>9} {'null_1%':>8} {'power':>6} {'overlap':>8}")
    rowsC = []
    for f in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        at = run_trials(T, a.seed + 4000 + int(f * 100), K=a.K, B=B, L=a.L, f=f, attack=True, modes=modes)
        nu = run_trials(T, a.seed + 5000 + int(f * 100), K=a.K, B=B, L=a.L, f=f, attack=False, modes=modes)
        th = np.quantile(nu["lam_min"], 0.01)
        pw = (at["lam_min"] < th).mean()
        print(f"  {f:5.2f} {ms(at['lam_min']):>17} {at['lam_pred'].mean():9.4f} {th:8.4f} {pw:6.2f} {at['overlap'].mean():8.3f}")
        rowsC.append([f, float(at["lam_min"].mean()), float(at["lam_pred"].mean()), float(th), float(pw), float(at["overlap"].mean())])
    res["C"] = {"fstar_pred": float(fstar), "rows": rowsC}

    # ---------------- Part D: K-independence (P5)
    print("\n[D] Vary number of Sybil accounts K (A fixed, f=0.3)  (P5)")
    print(f"  {'K':>4} {'lam_min':>17} {'lam_pred':>9} {'null_1%':>8} {'power':>6} {'overlap':>8} {'pairSNR':>8}")
    rowsD = []
    for K in [5, 10, 20, 40, 80]:
        at = run_trials(T, a.seed + 6000 + K, K=K, B=a.A - K, L=a.L, f=0.3, attack=True, modes=modes)
        nu = run_trials(T, a.seed + 7000 + K, K=K, B=a.A - K, L=a.L, f=0.3, attack=False, modes=modes)
        th = np.quantile(nu["lam_min"], 0.01)
        pw = (at["lam_min"] < th).mean()
        psnr = abs(np.nanmean(at["col_pair_mean"])) / at["ben_pair_sd"].mean()
        print(f"  {K:4d} {ms(at['lam_min']):>17} {at['lam_pred'].mean():9.4f} {th:8.4f} {pw:6.2f} {at['overlap'].mean():8.3f} {psnr:8.2f}")
        rowsD.append([K, float(at["lam_min"].mean()), float(at["lam_pred"].mean()), float(th), float(pw), float(at["overlap"].mean()), float(psnr)])
    res["D"] = rowsD

    with open("step1_results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nSaved step1_results.json   |   runtime {time.time()-t0:.1f}s")

    # ---------------- optional figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(a.seed + 9)
        acc, is_col, camp = build_world(a.K, B, a.L, 0.0, rng)
        ev = np.linalg.eigvalsh(cross_matrix(acc, modes))
        lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
        x = np.linspace(lo, hi, 400)
        mp = np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * np.pi * g * x)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(ev, bins=40, density=True, alpha=.6, label="eigenvalues of C")
        ax[0].plot(x, mp, "k-", label="Marchenko-Pastur (null)")
        ax[0].axvline(ev[0], color="r", ls="--", label=f"lam_min={ev[0]:.3f}")
        ax[0].set_title("Missing mode below the bulk"); ax[0].legend()
        ax[1].plot(rmag, S_of(camp, modes), ".", ms=3, label="union of colluders")
        ax[1].plot(rmag, np.mean([S_of(z, modes) for z in acc[:a.K]], 0), ".", ms=3, label="single colluder (avg)")
        ax[1].plot(rmag, S_of(np.vstack(acc[a.K:2 * a.K]), modes), ".", ms=3, label="union of benign")
        ax[1].set_xlabel("|n|"); ax[1].set_ylabel("S(k)"); ax[1].legend(); ax[1].set_title("Structure factors")
        plt.tight_layout(); plt.savefig("step1_figure.png", dpi=140)
        print("Saved step1_figure.png")
    except Exception as e:
        print(f"(figure skipped: {e})")

if __name__ == "__main__":
    main()
