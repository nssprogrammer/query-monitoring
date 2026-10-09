# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Paper numbers reproduced by this script
# Paper: "The Limits of Query Monitoring for Distributed Model Extraction".
# Run on Kaggle (see "Kaggle setup:" line below for hardware/time). Seeds are fixed in the configuration cell;
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Prediction log (Step 1), row "Union S(k) matches jittered-grid theory" (P1) | max relative error of the colluders' union structure factor vs jittered-grid theory | Pass (≤ 2% error) | Part A cell (shell table); VERDICT REPORT cell, `relA`, print "[P1] max rel err union vs theory" |
# | Prediction log (Step 1), row "Colluder pair correlation −(1 − S̄)/K" (P2) | mean colluder-pair C_ab, measured vs predicted | Pass (−0.0451 vs −0.0451) | Part B cell (`pred_pair`); VERDICT REPORT cell, print "[P2] pair mean=... pred=..." |
# | Prediction log (Step 1), row "BBP outlier below the bulk" (P3); App. E "Spectral detection condition" ("on a 2-D torus the measured lower outlier matched the BBP prediction to four decimals") | attacked λ_min, measured vs BBP prediction | Pass (0.0702 vs 0.0701) | Part B cell (`att["lam_min"]`, `att["lam_pred"]`); VERDICT REPORT cell, print "[P3] lam_min=... pred=..." |
# | Prediction log (Step 1), row "Padding threshold f∗ = 0.445" (P4) | predicted padding threshold f* vs empirical f at power 0.5 | 0.445 predicted; Partial (0.408) | Part C cell (`fstar`); VERDICT REPORT cell, `fc = interp_cross(...)`, print "[P4] empirical f(power=0.5)" |
# | Prediction log (Step 1), row "Outlier independent of K" (P5) | spread (max − min) of attacked λ_min over K ∈ {5, 10, 20, 40, 80} at f = 0.3 | Pass (spread 0.0017) | Part D cell; VERDICT REPORT cell, print "[P5] spread(max-min)=" |
#
# The P4 shortfall (0.408 vs 0.445) is the finite-size effect that Step 2 corrects with Tracy–Widom
# thresholds (Prediction log, Step 1 consequence "TW finite-size correction").
# A standalone CPU version of the same experiment is `step01_missing_mode_toy.py`.
#
# Not reported in the paper: per-shell S(k) values and benign-union S(k) (Part A table), λ_max attack vs null,
# eigenvector overlap and IPR, single-pair SNR, detection power at 1% FPR, the full power-vs-f table (Part C),
# per-K power/overlap/pair SNR (Part D), the GPU/numpy self-test, step1_figure.png and step1_results.json.

# %% [markdown]
# # Step 1 — The "missing mode": toy-world validation
# **Project:** Detecting Sybil-distributed model extraction via hyperuniformity of pooled query coverage.
#
# **Kaggle setup:** *Settings → Accelerator → GPU (T4 or P100)*, then **Run All**. The notebook falls back to CPU automatically (slower, fewer trials).
#
# **World.** Unit torus $[0,1)^2$. An attacker runs a space-filling (jittered-grid, hyperuniform) campaign of $L^2$ queries, randomly split over $K$ Sybil accounts, optionally padded with a fraction $f$ of random cover queries. $B$ benign accounts query uniformly at random. For each account we take Fourier amplitudes $\rho_a(\mathbf k)/\sqrt{n_a}$ on $m$ low-$k$ modes and form $C = FF^\*/m$, which is complex-Wishart (Marchenko–Pastur, ratio $\gamma=A/m$) under the null.
#
# | Prediction | Statement |
# |---|---|
# | **P1** | Union $S(k)\approx 1-\mathrm{sinc}^2(n_x/L)\,\mathrm{sinc}^2(n_y/L)$; single account $S\approx 1-(1-S)/K$ |
# | **P2** | Colluder-pair $C_{ab}\approx-(1-\bar S)/K$ — weak, pairwise-invisible |
# | **P3** | One eigenvalue escapes **below** $(1-\sqrt\gamma)^2$ at $(1+\ell)(1+\gamma/\ell)$, $\ell=\bar S-1$; $\lambda_{\max}$ unchanged |
# | **P4** | Padding threshold $f^\*=1-\sqrt\gamma/(1-\bar S_{\rm design})$ |
# | **P5** | Outlier location ≈ independent of $K$ |
#
# When finished, **copy the full output of the final cell (VERDICT REPORT)** plus any errors back into the chat.

# %%
# ===== Cell 1: configuration & backend =====
import math, json, time, os
import numpy as np

QUICK = False          # True -> smoke test (few trials)
SEED  = 12345
L     = 64             # campaign = L*L queries
NMAX  = 16             # |n| <= NMAX  -> m low-k modes
K     = 20             # Sybil accounts
A     = 100            # total accounts (K colluders + B benign)
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
K_GRID = [5, 10, 20, 40, 80]
F_FOR_K_SWEEP = 0.3
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

try:
    import torch
    HAS_TORCH = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
TRIALS = 10 if QUICK else (100 if USE_GPU else 30)
CHUNK = 65536 if USE_GPU else 8192

print(f"numpy {np.__version__} | torch: {torch.__version__ if HAS_TORCH else 'absent'}")
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU/numpy'} | TRIALS={TRIALS} | QUICK={QUICK}")
print(f"output dir: {OUTDIR}")


# %%
# ===== Cell 2: core functions =====
def half_plane_modes(nmax):
    # integer wavevectors 0<|n|<=nmax, one representative per +/- pair
    out = [(nx, ny) for nx in range(-nmax, nmax + 1) for ny in range(0, nmax + 1)
           if 0 < nx*nx + ny*ny <= nmax*nmax and not (ny == 0 and nx < 0)]
    return np.array(out, dtype=np.float64)

MODES = half_plane_modes(NMAX)
m = len(MODES)
RMAG = np.linalg.norm(MODES, axis=1)

def S_theory_jitter(modes, L):
    return 1.0 - (np.sinc(modes[:, 0] / L) * np.sinc(modes[:, 1] / L)) ** 2

def _fourier_numpy(pts, labels, nacc, modes):
    F = np.zeros((nacc, len(modes)), dtype=np.complex128)
    for s in range(0, len(pts), CHUNK):
        P, lab = pts[s:s+CHUNK], labels[s:s+CHUNK]
        E = np.exp(-2j * np.pi * (P @ modes.T))
        onehot = np.zeros((nacc, len(P)))
        onehot[lab, np.arange(len(P))] = 1.0
        F += onehot @ E
    return F

def _fourier_torch(pts, labels, nacc, modes):
    Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
    Fr = torch.zeros((nacc, len(modes)), dtype=torch.float64, device=DEV)
    Fi = torch.zeros_like(Fr)
    for s in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s:s+CHUNK], dtype=torch.float64, device=DEV)
        idx = torch.as_tensor(labels[s:s+CHUNK], dtype=torch.long, device=DEV)
        th = (2.0 * math.pi) * (P @ Mt.T)
        Fr.index_add_(0, idx, torch.cos(th))
        Fi.index_add_(0, idx, -torch.sin(th))
    return Fr.cpu().numpy() + 1j * Fi.cpu().numpy()

def fourier_accounts(pts, labels, nacc, modes=MODES):
    # raw rho_a(k) = sum_{j in a} exp(-2 pi i n.x_j)  -> (nacc, m)
    return _fourier_torch(pts, labels, nacc, modes) if USE_GPU else _fourier_numpy(pts, labels, nacc, modes)

def build_world(K, A, L, f_pad, seed, attack=True):
    # returns pts (N,2), labels (N,) in [0,A), counts (A,). Colluders are labels 0..K-1.
    rng = np.random.default_rng(seed)
    if attack:
        i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        design = (np.stack([i.ravel(), j.ravel()], 1) + rng.random((L*L, 2))) / L
    else:
        design = rng.random((L*L, 2))
    N0 = len(design)
    n_pad = int(round(f_pad / (1 - f_pad) * N0)) if f_pad > 0 else 0
    camp = np.vstack([design, rng.random((n_pad, 2))])
    camp = camp[rng.permutation(len(camp))]
    lab_c = rng.integers(0, K, len(camp))
    B = A - K
    n_per = len(camp) // K
    ben = rng.random((B * n_per, 2))
    lab_b = K + np.repeat(np.arange(B), n_per)
    pts = np.vstack([camp, ben]); labels = np.concatenate([lab_c, lab_b])
    return pts, labels, np.bincount(labels, minlength=A)

def analyse(Fraw, counts, K):
    A_ = len(counts); g = A_ / m
    F = Fraw / np.sqrt(counts)[:, None]
    Cm = F @ F.conj().T / m
    ev, V = np.linalg.eigh(Cm)
    is_col = np.arange(A_) < K
    w = np.where(is_col, np.sqrt(counts), 0.0); w /= np.linalg.norm(w)
    v = V[:, 0]
    S_union = np.abs(Fraw[:K].sum(0)) ** 2 / counts[:K].sum()
    S_bar = S_union.mean(); l = S_bar - 1
    lam_pred = (1 + l) * (1 + g / l) if l < -math.sqrt(g) else (1 - math.sqrt(g)) ** 2
    off = lambda Mx: Mx[~np.eye(len(Mx), dtype=bool)]
    cc = Cm[np.ix_(is_col, is_col)].real; bb = Cm[np.ix_(~is_col, ~is_col)].real
    return dict(gamma=g, lam_min=ev[0], lam_2=ev[1], lam_max=ev[-1],
                lo_edge=(1 - math.sqrt(g)) ** 2, hi_edge=(1 + math.sqrt(g)) ** 2,
                lam_pred=lam_pred, S_bar_union=S_bar,
                overlap=abs(np.vdot(v, w)) ** 2, ipr=float(np.sum(np.abs(v) ** 4)),
                col_pair_mean=off(cc).mean() if K > 1 else np.nan,
                ben_pair_mean=off(bb).mean(), ben_pair_sd=off(bb).std())

def run_trials(T, seed, K, A, L, f, attack):
    rows = []
    for t in range(T):
        pts, lab, cnt = build_world(K, A, L, f, seed + t, attack)
        rows.append(analyse(fourier_accounts(pts, lab, A), cnt, K))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

ms = lambda x: f"{np.nanmean(x):+.4f} ± {np.nanstd(x):.4f}"
RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, L=L, NMAX=NMAX, m=m, K=K, A=A,
                          TRIALS=TRIALS, backend="gpu" if USE_GPU else "cpu")}
print(f"m = {m} modes | gamma = A/m = {A/m:.3f} | lower MP edge = {(1-math.sqrt(A/m))**2:.4f}")

# quick self-test: GPU and numpy paths must agree (only runs if GPU present)
if USE_GPU:
    pts, lab, cnt = build_world(K, A, L, 0.0, 999, True)
    d = np.abs(_fourier_torch(pts, lab, A, MODES) - _fourier_numpy(pts, lab, A, MODES)).max()
    print(f"GPU vs numpy max |diff| = {d:.2e}  (should be < 1e-6)")

# %% [markdown]
# ## Part A — structure factors (P1)

# %%
t0 = time.time()
B = A - K
Sth = S_theory_jitter(MODES, L)
Su, Sacc, Sben = [], [], []
for t in range(TRIALS):
    pts, lab, cnt = build_world(K, A, L, 0.0, SEED + 1000 + t, True)
    Fr = fourier_accounts(pts, lab, A)
    Su.append(np.abs(Fr[:K].sum(0)) ** 2 / cnt[:K].sum())
    Sacc.append((np.abs(Fr[:K]) ** 2 / cnt[:K, None]).mean(0))
    Sben.append(np.abs(Fr[K:2*K].sum(0)) ** 2 / cnt[K:2*K].sum())
Su, Sacc, Sben = (np.mean(z, 0) for z in (Su, Sacc, Sben))
bins = [1, 4, 8, 12, NMAX + 1e-9]
print(f"{'shell':>10} {'S_union':>9} {'theory':>9} {'S_1acct':>9} {'pred_1acct':>11} {'S_benign_union':>15}")
rowsA = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (RMAG >= lo) & (RMAG < hi)
    pred1 = 1 - (1 - Sth[s].mean()) / K
    rowsA.append(dict(lo=lo, hi=hi, S_union=Su[s].mean(), theory=Sth[s].mean(),
                      S_1acct=Sacc[s].mean(), pred_1acct=pred1, S_benign=Sben[s].mean()))
    r = rowsA[-1]
    print(f"[{lo:>2},{hi:>5.0f}) {r['S_union']:9.4f} {r['theory']:9.4f} {r['S_1acct']:9.4f} {pred1:11.4f} {r['S_benign']:15.4f}")
RESULTS["A"] = rowsA
print(f"Part A done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part B — spectrum at f = 0 (P2, P3)

# %%
t0 = time.time()
att = run_trials(TRIALS, SEED + 2000, K, A, L, 0.0, True)
nul = run_trials(TRIALS, SEED + 3000, K, A, L, 0.0, False)
thr = np.quantile(nul["lam_min"], 0.01)
pred_pair = -(1 - att["S_bar_union"].mean()) / K
pair_snr = abs(att["col_pair_mean"].mean()) / att["ben_pair_sd"].mean()
power0 = (att["lam_min"] < thr).mean()
print(f"MP lower edge                  : {att['lo_edge'][0]:.4f}")
print(f"MP upper edge                  : {att['hi_edge'][0]:.4f}")
print(f"S_bar(union of campaign)       : {ms(att['S_bar_union'])}")
print(f"lam_min ATTACK measured        : {ms(att['lam_min'])}")
print(f"lam_min ATTACK BBP prediction  : {ms(att['lam_pred'])}")
print(f"lam_2   ATTACK (expect ~edge)  : {ms(att['lam_2'])}")
print(f"lam_min NULL                   : {ms(nul['lam_min'])}   1%-quantile = {thr:.4f}")
print(f"lam_max ATTACK vs NULL         : {ms(att['lam_max'])}  vs  {ms(nul['lam_max'])}")
print(f"overlap |<v_min,w>|^2          : {ms(att['overlap'])}   (null {ms(nul['overlap'])})")
print(f"IPR(v_min)  [1/K={1/K:.3f}]       : {ms(att['ipr'])}")
print(f"colluder-pair mean C_ab        : {ms(att['col_pair_mean'])}   pred {pred_pair:+.4f}")
print(f"benign-pair mean C_ab          : {ms(att['ben_pair_mean'])}   single-pair sd {att['ben_pair_sd'].mean():.4f}")
print(f"single-pair SNR                : {pair_snr:.2f}")
print(f"detection power @1% FPR        : {power0:.2f}")
RESULTS["B"] = {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in att.items()}
RESULTS["B"].update(null_lam_min=[float(nul['lam_min'].mean()), float(nul['lam_min'].std())],
                    null_lam_max=[float(nul['lam_max'].mean()), float(nul['lam_max'].std())],
                    null_thr=float(thr), pair_pred=float(pred_pair), pair_snr=float(pair_snr), power=float(power0))
print(f"Part B done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part C — padding sweep (P4)

# %%
t0 = time.time()
S_design = att["S_bar_union"].mean(); g = A / m
# [PAPER] Prediction log (Step 1, P4): predicted padding threshold f* = 0.445
fstar = 1 - math.sqrt(g) / (1 - S_design)
print(f"predicted f* = {fstar:.3f}")
print(f"{'f':>5} {'lam_min':>17} {'lam_pred':>9} {'null_1%':>8} {'power':>6} {'overlap':>8}")
rowsC = []
for f in F_GRID:
    at = run_trials(TRIALS, SEED + 4000 + int(f*1000), K, A, L, f, True)
    nu = run_trials(TRIALS, SEED + 5000 + int(f*1000), K, A, L, f, False)
    th = np.quantile(nu["lam_min"], 0.01); pw = (at["lam_min"] < th).mean()
    rowsC.append(dict(f=f, lam_min=float(at["lam_min"].mean()), lam_min_sd=float(at["lam_min"].std()),
                      lam_pred=float(at["lam_pred"].mean()), null_thr=float(th), power=float(pw),
                      overlap=float(at["overlap"].mean())))
    print(f"{f:5.2f} {ms(at['lam_min']):>17} {at['lam_pred'].mean():9.4f} {th:8.4f} {pw:6.2f} {at['overlap'].mean():8.3f}")
RESULTS["C"] = dict(fstar_pred=float(fstar), rows=rowsC)
print(f"Part C done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part D — number of Sybil accounts K (P5)

# %%
t0 = time.time()
print(f"f = {F_FOR_K_SWEEP}, A fixed = {A}")
print(f"{'K':>4} {'lam_min':>17} {'lam_pred':>9} {'null_1%':>8} {'power':>6} {'overlap':>8} {'pairSNR':>8} {'n/acct':>7}")
rowsD = []
for Kk in K_GRID:
    at = run_trials(TRIALS, SEED + 6000 + Kk, Kk, A, L, F_FOR_K_SWEEP, True)
    nu = run_trials(TRIALS, SEED + 7000 + Kk, Kk, A, L, F_FOR_K_SWEEP, False)
    th = np.quantile(nu["lam_min"], 0.01); pw = (at["lam_min"] < th).mean()
    psnr = abs(np.nanmean(at["col_pair_mean"])) / at["ben_pair_sd"].mean()
    nper = int(round(L*L / (1 - F_FOR_K_SWEEP) / Kk))
    rowsD.append(dict(K=Kk, lam_min=float(at["lam_min"].mean()), lam_min_sd=float(at["lam_min"].std()),
                      lam_pred=float(at["lam_pred"].mean()), null_thr=float(th), power=float(pw),
                      overlap=float(at["overlap"].mean()), pair_snr=float(psnr), n_per_acct=nper))
    print(f"{Kk:4d} {ms(at['lam_min']):>17} {at['lam_pred'].mean():9.4f} {th:8.4f} {pw:6.2f} {at['overlap'].mean():8.3f} {psnr:8.2f} {nper:7d}")
RESULTS["D"] = rowsD
print(f"Part D done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
g = A / m
pts, lab, cnt = build_world(K, A, L, 0.0, SEED + 9, True)
Fr = fourier_accounts(pts, lab, A); F = Fr / np.sqrt(cnt)[:, None]
ev = np.linalg.eigvalsh(F @ F.conj().T / m)
lo, hi = (1 - math.sqrt(g)) ** 2, (1 + math.sqrt(g)) ** 2
x = np.linspace(lo, hi, 400)
mp = np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * math.pi * g * x)

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0,0].hist(ev, bins=40, density=True, alpha=.6, label="eigenvalues of C (one attack world)")
ax[0,0].plot(x, mp, "k-", label="Marchenko–Pastur null")
ax[0,0].axvline(ev[0], color="r", ls="--", label=f"λ_min = {ev[0]:.3f}")
ax[0,0].axvline(lo, color="gray", ls=":", label=f"lower edge = {lo:.3f}")
ax[0,0].set_title("P3: missing mode below the bulk"); ax[0,0].legend(fontsize=8)

ax[0,1].plot(RMAG, np.abs(Fr[:K].sum(0))**2/cnt[:K].sum(), ".", ms=3, label="union of colluders")
ax[0,1].plot(RMAG, (np.abs(Fr[:K])**2/cnt[:K,None]).mean(0), ".", ms=3, label="single colluder (avg)")
ax[0,1].plot(RMAG, np.abs(Fr[K:2*K].sum(0))**2/cnt[K:2*K].sum(), ".", ms=3, label="union of K benign")
ax[0,1].plot(np.sort(RMAG), S_theory_jitter(MODES, L)[np.argsort(RMAG)], "k-", lw=1, label="theory (union)")
ax[0,1].set_xlabel("|n|"); ax[0,1].set_ylabel("S(k)"); ax[0,1].set_title("P1: structure factors"); ax[0,1].legend(fontsize=8)

fC = [r["f"] for r in RESULTS["C"]["rows"]]
ax[1,0].plot(fC, [r["power"] for r in RESULTS["C"]["rows"]], "o-", label="power @1% FPR")
ax[1,0].axvline(RESULTS["C"]["fstar_pred"], color="r", ls="--", label=f"predicted f* = {RESULTS['C']['fstar_pred']:.2f}")
ax[1,0].set_xlabel("padding fraction f"); ax[1,0].set_ylabel("power"); ax[1,0].set_title("P4: hiding threshold"); ax[1,0].legend()

KD = [r["K"] for r in RESULTS["D"]]
ax[1,1].errorbar(KD, [r["lam_min"] for r in RESULTS["D"]], yerr=[r["lam_min_sd"] for r in RESULTS["D"]], fmt="o-", label="λ_min (attack)")
ax[1,1].plot(KD, [r["lam_pred"] for r in RESULTS["D"]], "s--", label="BBP prediction")
ax[1,1].plot(KD, [r["null_thr"] for r in RESULTS["D"]], "k:", label="null 1% threshold")
ax[1,1].set_xscale("log"); ax[1,1].set_xlabel("K (Sybil accounts)"); ax[1,1].set_title(f"P5: K-independence (f={F_FOR_K_SWEEP})"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step1_figure.png", dpi=140); plt.show()


# %% [markdown]
# ## VERDICT REPORT — copy this cell's output into the chat

# %%
def interp_cross(xs, ys, level=0.5):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")

R = RESULTS; Bq = R["B"]
print("=" * 70); print("STEP 1 VERDICT REPORT"); print("=" * 70)
print("config:", json.dumps(R["config"]))
# [PAPER] Prediction log (Step 1, P1): union S(k) vs jittered-grid theory, Pass (≤ 2% error)
relA = max(abs(r["S_union"] - r["theory"]) / r["theory"] for r in R["A"])
devA = max(abs(r["S_1acct"] - r["pred_1acct"]) for r in R["A"])
benA = max(abs(r["S_benign"] - 1) for r in R["A"])
print(f"[P1] max rel err union vs theory = {relA:.3f} | max |S_1acct - pred| = {devA:.4f} | max |S_benign-1| = {benA:.4f}")
# [PAPER] Prediction log (Step 1, P3) and App. E: BBP outlier below the bulk, measured 0.0702 vs predicted 0.0701
lm, lp, le = Bq["lam_min"][0], Bq["lam_pred"][0], Bq["lo_edge"][0]
print(f"[P3] lam_min={lm:.4f}  pred={lp:.4f}  rel.err={(lm-lp)/lp:+.3f}  lower_edge={le:.4f}  below_edge={lm<le}")
dmax = (Bq["lam_max"][0] - Bq["null_lam_max"][0]) / max(Bq["null_lam_max"][1], 1e-9)
print(f"[P3] lam_max attack - null = {dmax:+.2f} null-sd  (expect ~0) | overlap={Bq['overlap'][0]:.3f} | power={Bq['power']:.2f}")
# [PAPER] Prediction log (Step 1, P2): colluder-pair correlation, −0.0451 vs −0.0451
print(f"[P2] pair mean={Bq['col_pair_mean'][0]:+.4f}  pred={Bq['pair_pred']:+.4f}  single-pair SNR={Bq['pair_snr']:.2f}")
# [PAPER] Prediction log (Step 1, P4): empirical f at power 0.5, Partial (0.408) vs predicted 0.445
fc = interp_cross([r["f"] for r in R["C"]["rows"]], [r["power"] for r in R["C"]["rows"]])
print(f"[P4] empirical f(power=0.5) = {fc:.3f}   predicted f* = {R['C']['fstar_pred']:.3f}")
print("[P4] f -> power: " + ", ".join(f"{r['f']:.2f}:{r['power']:.2f}" for r in R["C"]["rows"]))
# [PAPER] Prediction log (Step 1, P5): λ_min spread across K, Pass (spread 0.0017)
lmK = [r["lam_min"] for r in R["D"]]
gapK = np.mean([r["null_thr"] - r["lam_min"] for r in R["D"]])
print(f"[P5] lam_min across K: " + ", ".join(f"K={r['K']}:{r['lam_min']:.4f}(pw {r['power']:.2f}, ov {r['overlap']:.2f})" for r in R["D"]))
print(f"[P5] spread(max-min)={max(lmK)-min(lmK):.4f} vs mean gap to null threshold={gapK:.4f}")
print("=" * 70)
with open(f"{OUTDIR}/step1_results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step1_results.json and {OUTDIR}/step1_figure.png")
