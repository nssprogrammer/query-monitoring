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
# Run on Kaggle (see "Kaggle:" line below for hardware/time). Seeds are fixed in the configuration cell;
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Prediction log (Step 2), row "Tracy–Widom calibration to 10−3"; App. E "Spectral detection condition" ("over 20,000 null worlds analytic Tracy–Widom thresholds ... matched false-alarm rates from 10−1 to 10−4") | false-alarm rate of analytic TW thresholds on λ_min / λ_max, α = 0.1 ... 0.0001, N_NULL = 20,000 null worlds (GPU) | Pass (to 10−4); "from 10−1 to 10−4" | Part A cell, `rowsA` loop over alpha; VERDICT REPORT print "[A] FPR (min / max) at alpha" |
# | Prediction log (Step 2), row "Raw fails, whitened calibrated under non-uniform traffic" | false alarms of the raw vs intensity-whitened statistic on a non-uniform benign null (A3) | Pass (100% vs 0.53%) | Part A cell, A3 prints "RAW" / "WHITENED"; VERDICT REPORT print "[A3]" |
# | Prediction log (Step 2), row "TW-corrected f50 at unseen sizes" | gap between measured and TW-predicted f50 across L, γ | Pass (≤ 0.033) | Part B cell, print "f50 meas=... pred=..."; VERDICT REPORT "[B]" lines (max gap not isolated) |
# | Prediction log (Step 2), row "Optimal window peak" | predicted vs measured optimal window n_max (C1) | Pass (37 vs 36) | Part C cell, print "predicted optimal nmax = ...; measured best = ..." |
# | Prediction log (Step 2), row "N∗ ∝ A" | log-log slope of critical campaign size N* vs A (C2) | Pass (slope 0.95) | Part C cell, `slope_m`, print "log-log slope N* vs A: measured"; VERDICT REPORT "[C2] log-log slope" |
# | Prediction log (Step 2), row "Same-marginal stratified attacker detectable" | whitened clean-λ_min rate for S2 `warped_jitter` (Part D) | Pass (1.00) | Part D cell, row "S2 warped_jitter", column WHT cln; VERDICT REPORT "[D] S2 warped_jitter" |
#
# Consequences recorded in Prediction log: analytic thresholds adopted; whitening adopted.
#
# Not reported in the paper: null λ_min/λ_max moments vs TW, heterogeneous-size FPR (A2), the eigenvector-overlap
# formula check, per-configuration f50 and FPR checks, per-n_max f50 (C1), per-A N* values (C2), the other
# zoo designs (Part D), step2_figure.png and step2_results.json.

# %% [markdown]
# # Step 2 — Theory hardening, scaling law, and attacker-design zoo
# **Kaggle:** *Settings → Accelerator → GPU*, then **Run All** (≈40–70 min on T4/P100). Set `QUICK=True` for a 2–3 min smoke test.
#
# Step 1 confirmed the mechanism. Step 2 turns it into **quantitative, prospectively-tested theory** and probes **which attacker designs** create the signal.
#
# | Part | Question | Locked prediction (written before running) |
# |---|---|---|
# | **A** | Can the false-positive rate be set *analytically* (no null simulation)? | Null $\lambda_{\min}$ follows Tracy–Widom($\beta$=2) at the lower Marchenko–Pastur edge; empirical FPR ≈ nominal at α = 1e-2, 1e-3 (incl. heterogeneous account sizes) |
# | **B** | Does the finite-size threshold law hold **across sizes it was not fitted on**? | Empirical $f_{50}$ ≈ TW-corrected $f_{50}$ (printed *before* simulation), gap shrinks with L; overlap formula $(1-\gamma/\ell^2)/(1+\gamma/\ell)$ holds away from the transition |
# | **C1** | Optimal spectral window for the defender | Measured hiding budget $f_{50}(n_{\max})$ tracks the predicted curve and peaks near the predicted optimum |
# | **C2** | Scaling law: how big must a campaign be to be visible? | Critical campaign size $N^\* \propto A$ (log–log slope ≈ 1) |
# | **A3** | Does non-uniform benign traffic break the null? | RAW statistic: false positives. Intensity-**whitened** statistic: calibrated (~nominal) |
# | **D** | Which attacker designs produce the signal, and does it survive non-uniform benign traffic? | See the locked table in Part D. Key claim: a stratified attacker whose query **marginal equals the benign distribution** still yields a clean λ_min outlier after whitening |
#
# At the end, copy the **VERDICT REPORT** output into the chat (plus the figure).

# %%
# ===== Cell 1: configuration & backend =====
import math, json, time, os
import numpy as np
from scipy.special import airy
from scipy.optimize import brentq
from scipy.stats import qmc
import scipy.sparse as sp

QUICK = False
SEED = 2026
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

try:
    import torch
    HAS_TORCH = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
CHUNK = 65536 if USE_GPU else 8192

if QUICK:
    N_NULL, N_NULL_HET, TR_B, TR_C, TR_D = 400, 200, 6, 5, 3
    B_LS, B_GAMMAS = [32, 64], [0.25]
    C1_NMAX = [18, 32, 48]; C2_AS = [50, 100]; C2_LS = [16, 24, 32, 48]
    N_A3, N_REF = 150, 200000
elif USE_GPU:
    N_NULL, N_NULL_HET, TR_B, TR_C, TR_D = 20000, 5000, 80, 50, 30
    B_LS, B_GAMMAS = [32, 64, 128], [0.1, 0.25, 0.5]
    C1_NMAX = [18, 24, 30, 36, 42, 48, 54, 60]; C2_AS = [50, 100, 200, 400]; C2_LS = [16, 20, 24, 28, 32, 40, 48, 56, 64]
    N_A3, N_REF = 3000, 2000000
else:  # CPU fallback: reduced
    N_NULL, N_NULL_HET, TR_B, TR_C, TR_D = 4000, 1000, 30, 20, 10
    B_LS, B_GAMMAS = [32, 64], [0.1, 0.25, 0.5]
    C1_NMAX = [18, 30, 42, 60]; C2_AS = [50, 100, 200]; C2_LS = [16, 24, 32, 40, 48, 64]
    N_A3, N_REF = 800, 1000000

BASE = dict(L=64, NMAX=16, K=20, A=100)
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU/numpy'} | QUICK={QUICK}")
print(f"N_NULL={N_NULL} TR_B={TR_B} TR_C={TR_C} TR_D={TR_D} | B_LS={B_LS} B_GAMMAS={B_GAMMAS}")


# %%
# ===== Cell 2: Tracy-Widom (beta=2) via Bornemann's Fredholm determinant =====
def tw2_cdf(s, n=60, Lw=14.0):
    x, w = np.polynomial.legendre.leggauss(n)
    x = s + (x + 1) * Lw / 2; w = w * Lw / 2
    ai, aip, _, _ = airy(x)
    X, Y = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        Kk = (ai[:, None] * aip[None, :] - aip[:, None] * ai[None, :]) / (X - Y)
    Kk[np.diag_indices(n)] = aip ** 2 - x * ai ** 2
    sw = np.sqrt(w)
    return float(np.linalg.det(np.eye(n) - sw[:, None] * Kk * sw[None, :]))

_TWQ = {}
def tw_q(p):
    if p not in _TWQ:
        _TWQ[p] = brentq(lambda v: tw2_cdf(v) - p, -9, 8)
    return _TWQ[p]
_s = np.linspace(-8, 6, 1401); _c = np.array([tw2_cdf(v) for v in _s]); _pdf = np.gradient(_c, _s)
TW_MEAN = float(np.trapezoid(_s * _pdf, _s)); TW_SD = float(np.sqrt(np.trapezoid((_s - TW_MEAN) ** 2 * _pdf, _s)))
TW_GRID, TW_PDF = _s, _pdf
print(f"TW2 mean={TW_MEAN:.4f} sd={TW_SD:.4f} q99={tw_q(0.99):.4f} q999={tw_q(0.999):.4f}  (ref: -1.7711, 0.9018, 0.4776, 1.3144)")

def edges(p, n):
    # C = F F*/n with F (p x n), unit-variance complex entries, p < n
    a, b = math.sqrt(n), math.sqrt(p)
    mu_m = (a - b) ** 2 / n; sg_m = (a - b) * (1 / b - 1 / a) ** (1 / 3) / n
    mu_p = (a + b) ** 2 / n; sg_p = (a + b) * (1 / b + 1 / a) ** (1 / 3) / n
    return mu_m, sg_m, mu_p, sg_p
def thr_min(p, n, alpha):
    mu, sg, _, _ = edges(p, n); return mu - sg * tw_q(1 - alpha)
def thr_max(p, n, alpha):
    _, _, mu, sg = edges(p, n); return mu + sg * tw_q(1 - alpha)
def bbp_low(l, g):   # outlier location for spike l in (-1, -sqrt g)
    return (1 + l) * (1 + g / l)
def overlap_pred(l, g):
    return max(0.0, (1 - g / l ** 2) / (1 + g / l)) if l < -math.sqrt(g) else 0.0
def f50_pred(Sd, g, lam_thr):
    # padding fraction at which BBP outlier sits exactly on the alpha-threshold
    if lam_thr <= 0 or -(1 - Sd) >= -math.sqrt(g) or bbp_low(-(1 - Sd), g) >= lam_thr: return 0.0
    lstar = brentq(lambda l: bbp_low(l, g) - lam_thr, -(1 - 1e-12), -math.sqrt(g) - 1e-12)
    return max(0.0, 1 + lstar / (1 - Sd))
def fstar_asym(Sd, g):
    return max(0.0, 1 - math.sqrt(g) / (1 - Sd))


# %%
# ===== Cell 3: core simulation =====
_MODES = {}
def half_plane_modes(nmax):
    if nmax not in _MODES:
        nx, ny = np.meshgrid(np.arange(-nmax, nmax + 1), np.arange(0, nmax + 1), indexing="ij")
        nx, ny = nx.ravel(), ny.ravel(); r2 = nx * nx + ny * ny
        keep = (r2 > 0) & (r2 <= nmax * nmax) & ~((ny == 0) & (nx < 0))
        _MODES[nmax] = np.stack([nx[keep], ny[keep]], 1).astype(np.float64)
    return _MODES[nmax]

def window_table(L, nm_hi):
    # m(nm) and mean theoretical S_design over |n|<=nm, nm = 0..nm_hi (jittered grid)
    mo = half_plane_modes(nm_hi); r = np.sqrt((mo ** 2).sum(1)); S = S_theory_jitter(mo, L)
    order = np.argsort(r); r, S = r[order], S[order]; cs = np.cumsum(S)
    ms_ = np.searchsorted(r, np.arange(nm_hi + 1) + 1e-9, side="right")
    Sd = np.where(ms_ > 0, cs[np.maximum(ms_ - 1, 0)] / np.maximum(ms_, 1), np.nan)
    return ms_, Sd

def S_theory_jitter(modes, L):
    return 1.0 - (np.sinc(modes[:, 0] / L) * np.sinc(modes[:, 1] / L)) ** 2

def _fourier_numpy(pts, labels, nacc, modes):
    F = np.zeros((nacc, len(modes)), dtype=np.complex128)
    for s in range(0, len(pts), CHUNK):
        P, lab = pts[s:s + CHUNK], labels[s:s + CHUNK]
        E = np.exp(-2j * np.pi * (P @ modes.T))
        oh = sp.csr_matrix((np.ones(len(P)), (lab, np.arange(len(P)))), shape=(nacc, len(P)))
        F += oh @ E
    return F

def _fourier_torch(pts, labels, nacc, modes):
    Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
    Fr = torch.zeros((nacc, len(modes)), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
    for s in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s:s + CHUNK], dtype=torch.float64, device=DEV)
        idx = torch.as_tensor(labels[s:s + CHUNK], dtype=torch.long, device=DEV)
        th = (2.0 * math.pi) * (P @ Mt.T)
        Fr.index_add_(0, idx, torch.cos(th)); Fi.index_add_(0, idx, -torch.sin(th))
    return Fr.cpu().numpy() + 1j * Fi.cpu().numpy()

def fourier_accounts(pts, labels, nacc, modes):
    return _fourier_torch(pts, labels, nacc, modes) if USE_GPU else _fourier_numpy(pts, labels, nacc, modes)

# ---------- attacker designs on the unit torus (all return N points) ----------
def d_jitter(L, rng):
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    return (np.stack([i.ravel(), j.ravel()], 1) + rng.random((L * L, 2))) / L
def d_random(N, rng): return rng.random((N, 2))
def d_sobol(N, rng):
    return qmc.Sobol(d=2, scramble=True, seed=int(rng.integers(2**31))).random(N)
def _tor_d2(P, x):
    d = np.abs(P - x); d = np.minimum(d, 1 - d); return (d * d).sum(1)
def _fps(pool, N, rng):
    sel = [int(rng.integers(len(pool)))]; d2 = _tor_d2(pool, pool[sel[0]])
    for _ in range(N - 1):
        i = int(np.argmax(d2)); sel.append(i); d2 = np.minimum(d2, _tor_d2(pool, pool[i]))
    return pool[np.array(sel)]
def boundary(x): return 0.5 + 0.2 * np.sin(2 * np.pi * x)
def d_kcenter(N, rng, pool_mult=10): return _fps(rng.random((pool_mult * N, 2)), N, rng)
def d_random_pool(N, rng, pool_mult=10):
    pool = rng.random((pool_mult * N, 2)); return pool[rng.choice(len(pool), N, replace=False)]
def d_uncertainty(N, rng, pool_mult=10):
    pool = rng.random((pool_mult * N, 2)); dist = np.abs(pool[:, 1] - boundary(pool[:, 0]))
    return pool[np.argsort(dist)[:N]]
def d_band_kcenter(N, rng, pool_mult=10, half=0.05):
    x = rng.random(pool_mult * N); y = (boundary(x) + half * (2 * rng.random(pool_mult * N) - 1)) % 1.0
    return _fps(np.stack([x, y], 1), N, rng)

# ---------- non-uniform benign intensity (scenario S2): 50% uniform + 50% truncated-normal band in y ----------
from scipy.stats import norm as _norm
BUMP_MU, BUMP_SD = 0.5, 0.08
def benign_uniform(n, rng): return rng.random((n, 2))
def benign_bump(n, rng):
    x = rng.random(n); y = rng.random(n); is_b = rng.random(n) < 0.5
    yb = rng.normal(BUMP_MU, BUMP_SD, is_b.sum())
    while ((yb < 0) | (yb >= 1)).any():
        bad = (yb < 0) | (yb >= 1); yb[bad] = rng.normal(BUMP_MU, BUMP_SD, bad.sum())
    y[is_b] = yb; return np.stack([x, y], 1)
_yy = np.linspace(0, 1, 20001)
_Z = _norm.cdf((1 - BUMP_MU) / BUMP_SD) - _norm.cdf(-BUMP_MU / BUMP_SD)
_G = 0.5 * _yy + 0.5 * (_norm.cdf((_yy - BUMP_MU) / BUMP_SD) - _norm.cdf(-BUMP_MU / BUMP_SD)) / _Z
def warp_to_bump(P):  # inverse-CDF warp in y: stratified uniform -> stratified bump-distributed
    return np.stack([P[:, 0], np.interp(P[:, 1], _G, _yy)], 1)

# ---------- worlds ----------
def build_world(design, K, A, f_pad, rng, benign_fn=benign_uniform):
    # design: campaign points; colluders labels 0..K-1, benign K..A-1 (equal size, drawn from benign_fn)
    N0 = len(design)
    n_pad = int(round(f_pad / (1 - f_pad) * N0)) if f_pad > 0 else 0
    camp = np.vstack([design, benign_fn(n_pad, rng)]); camp = camp[rng.permutation(len(camp))]
    lab_c = rng.integers(0, K, len(camp))
    n_per = max(1, len(camp) // K)
    ben = benign_fn((A - K) * n_per, rng); lab_b = K + np.repeat(np.arange(A - K), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b])
    return pts, lab, np.bincount(lab, minlength=A)

def null_world_hetero(A, rng, median=200, sigma=1.0, nmin=20):
    sizes = np.maximum(nmin, np.round(median * np.exp(sigma * rng.standard_normal(A)))).astype(int)
    lab = np.repeat(np.arange(A), sizes)
    return rng.random((len(lab), 2)), lab, sizes

def spectrum(pts, lab, cnt, modes, K=0, vectors=True):
    m = len(modes); A_ = len(cnt)
    Fr = fourier_accounts(pts, lab, A_, modes)
    F = Fr / np.sqrt(np.maximum(cnt, 1))[:, None]
    Cm = F @ F.conj().T / m
    out = {}
    if vectors and K > 0:
        ev, V = np.linalg.eigh(Cm)
        w = np.where(np.arange(A_) < K, np.sqrt(cnt), 0.0); w = w / np.linalg.norm(w)
        out.update(ov_min=abs(np.vdot(V[:, 0], w)) ** 2, ov_max=abs(np.vdot(V[:, -1], w)) ** 2)
        out["S_union"] = float((np.abs(Fr[:K].sum(0)) ** 2).mean() / cnt[:K].sum())
    else:
        ev = np.linalg.eigvalsh(Cm)
    out.update(lam_min=ev[0], lam_max=ev[-1])
    return out

# ---------- intensity-relative whitening from a historical benign reference sample ----------
def make_reference(benign_fn, modes, n_ref, seed):
    # mu(k) and Sigma^{-1/2}, Sigma(k,k') = mu(k-k') - mu(k) conj(mu(k'))  (iid-sampling null)
    rng = np.random.default_rng(seed); P = benign_fn(n_ref, rng); m = len(modes)
    G = np.zeros((m, m), dtype=np.complex128); mu = np.zeros(m, dtype=np.complex128)
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
        Gt = torch.zeros((m, m), dtype=torch.complex128, device=DEV); mut = torch.zeros(m, dtype=torch.complex128, device=DEV)
        for s0 in range(0, n_ref, 16384):
            th = (2 * math.pi) * (torch.as_tensor(P[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            E = torch.complex(torch.cos(th), -torch.sin(th))
            Gt += E.T @ E.conj(); mut += E.sum(0)
        G, mu = Gt.cpu().numpy(), mut.cpu().numpy()
    else:
        for s0 in range(0, n_ref, 4096):
            E = np.exp(-2j * np.pi * (P[s0:s0 + 4096] @ modes.T)); G += E.T @ E.conj(); mu += E.sum(0)
    G /= n_ref; mu /= n_ref
    Sig = G - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew, U = np.linalg.eigh(Sig); ew = np.maximum(ew, 1e-6 * ew.max())
    W = (U / np.sqrt(ew)) @ U.conj().T          # Sigma^{-1/2}
    return dict(mu=mu, WT=W.T, sig_eig_min=float(ew.min()), sig_eig_max=float(ew.max()))

def spectra_raw_white(pts, lab, cnt, modes, K, ref, tmin, tmax):
    m = len(modes); A_ = len(cnt)
    Fr = fourier_accounts(pts, lab, A_, modes); sq = np.sqrt(np.maximum(cnt, 1))
    w = np.where(np.arange(A_) < K, sq, 0.0); w = w / np.linalg.norm(w)
    out = {}
    Fraw = Fr / sq[:, None]
    Fw = ((Fr - cnt[:, None] * ref["mu"][None, :]) / sq[:, None]) @ ref["WT"]
    for tag, F in (("raw", Fraw), ("wht", Fw)):
        ev, V = np.linalg.eigh(F @ F.conj().T / m)
        nb, na = int((ev < tmin).sum()), int((ev > tmax).sum())
        out[f"{tag}_clean_min"] = float(nb == 1); out[f"{tag}_multi_min"] = float(nb >= 2)
        out[f"{tag}_max"] = float(na >= 1)
        out[f"{tag}_ov_min"] = abs(np.vdot(V[:, 0], w)) ** 2; out[f"{tag}_ov_max"] = abs(np.vdot(V[:, -1], w)) ** 2
        out[f"{tag}_lam_min"] = ev[0]; out[f"{tag}_lam_max"] = ev[-1]
    out["S_union_wht"] = float((np.abs((Fw[:K] * sq[:K, None]).sum(0)) ** 2).mean() / cnt[:K].sum())
    return out

def run(design_fn, K, A, f, modes, T, seed):
    rows = []
    for t in range(T):
        rng = np.random.default_rng(seed + t)
        pts, lab, cnt = build_world(design_fn(rng), K, A, f, rng)
        rows.append(spectrum(pts, lab, cnt, modes, K))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

def interp_cross(xs, ys, level=0.5):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")

def choose_K(A): return max(min(20, A // 2), A // 10)

RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, backend="gpu" if USE_GPU else "cpu", N_NULL=N_NULL,
                          TR_B=TR_B, TR_C=TR_C, TR_D=TR_D)}
if USE_GPU:
    mm = half_plane_modes(16); rng = np.random.default_rng(1)
    pts, lab, cnt = build_world(d_jitter(64, rng), 20, 100, 0.0, rng)
    d = np.abs(_fourier_torch(pts, lab, 100, mm) - _fourier_numpy(pts, lab, 100, mm)).max()
    print(f"GPU vs numpy max |diff| = {d:.2e}  (must be < 1e-6)"); RESULTS["gpu_check"] = float(d)
print("core ready")

# %% [markdown]
# ## Part A — analytic false-positive calibration (Tracy–Widom)

# %%
t0 = time.time()
L0, NM0, A0 = BASE["L"], BASE["NMAX"], BASE["A"]
modes0 = half_plane_modes(NM0); m0 = len(modes0)
mu_m, sg_m, mu_p, sg_p = edges(A0, m0)
lmin, lmax = [], []
for t in range(N_NULL):
    rng = np.random.default_rng(SEED + 10_000_000 + t)
    pts, lab, cnt = build_world(d_random(L0 * L0, rng), BASE["K"], A0, 0.0, rng)
    s = spectrum(pts, lab, cnt, modes0, vectors=False); lmin.append(s["lam_min"]); lmax.append(s["lam_max"])
lmin, lmax = np.array(lmin), np.array(lmax)
print(f"A={A0} m={m0} gamma={A0/m0:.4f}  N_NULL={N_NULL}")
print(f"lam_min: measured {lmin.mean():.5f} ± {lmin.std():.5f} | TW pred {mu_m - sg_m*TW_MEAN:.5f} ± {sg_m*TW_SD:.5f}")
print(f"lam_max: measured {lmax.mean():.5f} ± {lmax.std():.5f} | TW pred {mu_p + sg_p*TW_MEAN:.5f} ± {sg_p*TW_SD:.5f}")
rowsA = []
print(f"{'alpha':>8} {'thr_min':>9} {'FPR_min':>9} {'cnt':>5} {'thr_max':>9} {'FPR_max':>9} {'cnt':>5} {'expected':>8}")
# [PAPER] Prediction log (Step 2) and App. E: analytic TW thresholds over 20,000 null worlds, false alarms matched "from 10−1 to 10−4"
for a in [0.1, 0.05, 0.01, 0.001, 0.0001]:
    tm, tM = thr_min(A0, m0, a), thr_max(A0, m0, a)
    c1, c2 = int((lmin < tm).sum()), int((lmax > tM).sum())
    rowsA.append(dict(alpha=a, thr_min=tm, fpr_min=c1 / N_NULL, cnt_min=c1, thr_max=tM, fpr_max=c2 / N_NULL, cnt_max=c2))
    print(f"{a:8.4f} {tm:9.5f} {c1/N_NULL:9.5f} {c1:5d} {tM:9.5f} {c2/N_NULL:9.5f} {c2:5d} {a*N_NULL:8.1f}")
# A2: heterogeneous (log-normal) account sizes
lmin_h, lmax_h = [], []
for t in range(N_NULL_HET):
    rng = np.random.default_rng(SEED + 20_000_000 + t)
    pts, lab, cnt = null_world_hetero(A0, rng)
    s = spectrum(pts, lab, cnt, modes0, vectors=False); lmin_h.append(s["lam_min"]); lmax_h.append(s["lam_max"])
lmin_h, lmax_h = np.array(lmin_h), np.array(lmax_h)
rowsA2 = []
print(f"\nA2 heterogeneous sizes (lognormal, median 200, sigma 1, min 20), N={N_NULL_HET}")
for a in [0.05, 0.01, 0.001]:
    c1 = int((lmin_h < thr_min(A0, m0, a)).sum()); c2 = int((lmax_h > thr_max(A0, m0, a)).sum())
    rowsA2.append(dict(alpha=a, fpr_min=c1 / N_NULL_HET, fpr_max=c2 / N_NULL_HET))
    print(f"  alpha={a:6.3f}  FPR_min={c1/N_NULL_HET:.4f} ({c1})  FPR_max={c2/N_NULL_HET:.4f} ({c2})  expected {a*N_NULL_HET:.1f}")
# A3: NON-UNIFORM benign null (bump intensity): raw vs intensity-whitened statistic
REF_BUMP = make_reference(benign_bump, modes0, N_REF, SEED + 31337)
REF_UNIF = make_reference(benign_uniform, modes0, N_REF, SEED + 31338)
print(f"\nA3 reference Sigma eigen-range: bump [{REF_BUMP['sig_eig_min']:.3f}, {REF_BUMP['sig_eig_max']:.3f}]  uniform [{REF_UNIF['sig_eig_min']:.3f}, {REF_UNIF['sig_eig_max']:.3f}]")
tmA, tMA = thr_min(A0, m0, 0.005), thr_max(A0, m0, 0.005)
acc3 = []
for t in range(N_A3):
    rng = np.random.default_rng(SEED + 30_000_000 + t)
    pts, lab, cnt = build_world(benign_bump(L0 * L0, rng), BASE["K"], A0, 0.0, rng, benign_fn=benign_bump)
    acc3.append(spectra_raw_white(pts, lab, cnt, modes0, BASE["K"], REF_BUMP, tmA, tMA))
A3 = {k: float(np.mean([r[k] for r in acc3])) for k in ["raw_clean_min", "raw_multi_min", "raw_max", "wht_clean_min", "wht_multi_min", "wht_max"]}
# [PAPER] Prediction log (Step 2): raw vs whitened false alarms under non-uniform traffic, Pass (100% vs 0.53%)
print(f"A3 non-uniform null, N={N_A3}, N_REF={N_REF}, per-side alpha=0.005 (expect ~0.005 each when calibrated):")
print(f"   RAW      : any-min={A3['raw_clean_min']+A3['raw_multi_min']:.4f} (multi {A3['raw_multi_min']:.4f})  max={A3['raw_max']:.4f}")
print(f"   WHITENED : any-min={A3['wht_clean_min']+A3['wht_multi_min']:.4f} (multi {A3['wht_multi_min']:.4f})  max={A3['wht_max']:.4f}")
RESULTS["A"] = dict(A3=A3, m=m0, A=A0, lmin_mean=float(lmin.mean()), lmin_sd=float(lmin.std()),
                    lmin_pred=float(mu_m - sg_m * TW_MEAN), lmin_sd_pred=float(sg_m * TW_SD),
                    lmax_mean=float(lmax.mean()), lmax_pred=float(mu_p + sg_p * TW_MEAN), rows=rowsA, hetero=rowsA2)
NULL_LMIN = lmin
print(f"Part A done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part B — prospective finite-size threshold law (predictions printed first)

# %%
t0 = time.time()
ALPHA = 0.01
cfgs = []
for L in B_LS:
    nmax = L // 4; modes = half_plane_modes(nmax); m = len(modes)
    for g in B_GAMMAS:
        A = int(round(g * m)); K = choose_K(A); gg = A / m
        Sd = float(S_theory_jitter(modes, L).mean())
        lt = thr_min(A, m, ALPHA)
        cfgs.append(dict(L=L, nmax=nmax, m=m, A=A, K=K, gamma=gg, Sd=Sd, lam_thr=lt,
                         fstar=fstar_asym(Sd, gg), f50_pred=f50_pred(Sd, gg, lt)))
print("LOCKED PREDICTIONS (computed from theory only):")
print(f"{'L':>4} {'m':>5} {'A':>4} {'K':>3} {'gamma':>6} {'S_d':>6} {'f*_asym':>8} {'f50_TW':>7}")
for c in cfgs:
    print(f"{c['L']:4d} {c['m']:5d} {c['A']:4d} {c['K']:3d} {c['gamma']:6.3f} {c['Sd']:6.3f} {c['fstar']:8.3f} {c['f50_pred']:7.3f}")

print("\nSIMULATION:")
for c in cfgs:
    modes = half_plane_modes(c["nmax"]); T = TR_B if c["L"] < 128 else max(TR_B // 2, 5)
    fgrid = sorted(set(np.clip(np.round(c["f50_pred"] + np.array([-.15, -.1, -.05, 0, .05, .1, .15]), 3), 0, 0.95)))
    pw, ovm, ovp, lms = [], [], [], []
    for f in fgrid:
        r = run(lambda rg: d_jitter(c["L"], rg), c["K"], c["A"], f, modes, T, SEED + 100 * c["L"] + int(1000 * f) + int(1e4 * c["gamma"]))
        pw.append(float((r["lam_min"] < c["lam_thr"]).mean())); lms.append(float(r["lam_min"].mean()))
        l = -(1 - f) * (1 - c["Sd"]); ovm.append(float(r["ov_min"].mean())); ovp.append(overlap_pred(l, c["gamma"]))
    # empirical FPR of the analytic threshold in this config
    nfp = 0; NN = 100 if not QUICK else 20
    for t in range(NN):
        rng = np.random.default_rng(SEED + 7_000_000 + t + 1000 * c["L"] + int(1e4 * c["gamma"]))
        pts, lab, cnt = build_world(d_random(c["L"] ** 2, rng), c["K"], c["A"], 0.0, rng)
        nfp += spectrum(pts, lab, cnt, modes, vectors=False)["lam_min"] < c["lam_thr"]
    c.update(fgrid=[float(x) for x in fgrid], power=pw, lam_min=lms, ov_meas=ovm, ov_pred=ovp,
             f50_meas=interp_cross(fgrid, pw), fpr_check=nfp / NN, T=T)
    # [PAPER] Prediction log (Step 2): TW-corrected f50 at unseen sizes, |meas − pred| Pass (≤ 0.033)
    print(f"L={c['L']:3d} g={c['gamma']:.2f}: f50 meas={c['f50_meas']:.3f} pred={c['f50_pred']:.3f} asym={c['fstar']:.3f} "
          f"| FPR@thr={c['fpr_check']:.3f} | power: " + " ".join(f"{f:.2f}:{p:.2f}" for f, p in zip(fgrid, pw)))
RESULTS["B"] = cfgs
print(f"Part B done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part C — optimal window (C1) and campaign-size scaling law (C2)

# %%
t0 = time.time()
# ---- C1: hiding budget f50 vs window nmax at L=64, A=400
L1, A1 = 64, 400; K1 = choose_K(A1); rowsC1 = []
print(f"C1: L={L1} A={A1} K={K1}")
print(f"{'nmax':>5} {'m':>5} {'gamma':>6} {'S_d':>6} {'f50_pred':>8} {'f50_meas':>8}")
for nm in C1_NMAX:
    modes = half_plane_modes(nm); m = len(modes); g = A1 / m
    if g > 0.8: print(f"{nm:5d} {m:5d} {g:6.3f}  skipped (gamma>0.8)"); continue
    Sd = float(S_theory_jitter(modes, L1).mean()); lt = thr_min(A1, m, 0.01); fp = f50_pred(Sd, g, lt)
    fgrid = sorted(set(np.clip(np.round(fp + np.array([-.15, -.075, 0, .075, .15]), 3), 0, 0.95))); pw = []
    for f in fgrid:
        r = run(lambda rg: d_jitter(L1, rg), K1, A1, f, modes, TR_C, SEED + 31 * nm + int(1000 * f))
        pw.append(float((r["lam_min"] < lt).mean()))
    fm = interp_cross(fgrid, pw)
    rowsC1.append(dict(nmax=nm, m=m, gamma=g, Sd=Sd, f50_pred=fp, f50_meas=fm, fgrid=[float(x) for x in fgrid], power=pw))
    print(f"{nm:5d} {m:5d} {g:6.3f} {Sd:6.3f} {fp:8.3f} {fm:8.3f}")
# predicted optimum on a fine grid (nmax up to L-1; jittered-grid theory exact for |n|<L)
ms_t, Sd_t = window_table(L1, L1 - 1)
pred_curve = [(nm, f50_pred(Sd_t[nm], A1 / ms_t[nm], thr_min(A1, ms_t[nm], 0.01))) for nm in range(10, L1) if ms_t[nm] > 0 and A1 / ms_t[nm] <= 0.8]
best = max(pred_curve, key=lambda z: z[1])
meas_ok = [r for r in rowsC1 if r["f50_meas"] == r["f50_meas"]]
# [PAPER] Prediction log (Step 2): optimal window peak, Pass (37 vs 36)
print(f"predicted optimal nmax = {best[0]} (f50={best[1]:.3f}); measured best = {max(meas_ok, key=lambda r: r['f50_meas'])['nmax'] if meas_ok else 'n/a'}")

# ---- C2: critical campaign size N* vs number of monitored accounts A (f=0, optimal window)
def best_window(L, A):
    # defender's optimal window: maximise predicted hiding budget f50 (0 if undetectable at f=0)
    ms_t, Sd_t = window_table(L, L - 1); best_ = (None, -1.0)
    for nm in range(2, L):
        if ms_t[nm] == 0 or A / ms_t[nm] > 0.8: continue
        fp = f50_pred(Sd_t[nm], A / ms_t[nm], thr_min(A, ms_t[nm], 0.01))
        if fp > best_[1]: best_ = (nm, fp)
    return best_
rowsC2 = []
print(f"\nC2: power at f=0 (TW alpha=0.01) vs campaign side L (N=L^2); '*' = predicted detectable")
for A in C2_AS:
    K = min(20, A // 2); line = []
    for L in C2_LS:
        nm, fp = best_window(L, A)
        if nm is None: line.append((L, 0.0, False)); continue
        modes = half_plane_modes(nm); lt = thr_min(A, len(modes), 0.01)
        r = run(lambda rg: d_jitter(L, rg), K, A, 0.0, modes, TR_C, SEED + 7 * A + L)
        line.append((L, float((r["lam_min"] < lt).mean()), fp > 0))
    Lm = interp_cross([z[0] for z in line], [-z[1] for z in line], level=-0.5)   # power rising through 0.5
    Lp = next((L for L in range(8, 256) if best_window(L, A)[1] > 0), float("nan"))
    rowsC2.append(dict(A=A, line=line, Nstar_meas=Lm ** 2 if Lm == Lm else float("nan"), Nstar_pred=float(Lp) ** 2))
    print(f"A={A:4d}: " + " ".join(f"L{L}:{p:.2f}{'*' if d else ''}" for L, p, d in line) +
          f" | N* meas={rowsC2[-1]['Nstar_meas']:.0f} pred={rowsC2[-1]['Nstar_pred']:.0f}")
ok = [r for r in rowsC2 if r["Nstar_meas"] == r["Nstar_meas"]]
# [PAPER] Prediction log (Step 2): N* ∝ A, measured log-log slope 0.95
slope_m = np.polyfit(np.log([r["A"] for r in ok]), np.log([r["Nstar_meas"] for r in ok]), 1)[0] if len(ok) >= 2 else float('nan')
slope_p = np.polyfit(np.log([r["A"] for r in rowsC2]), np.log([r["Nstar_pred"] for r in rowsC2]), 1)[0] if len(rowsC2) >= 2 else float('nan')
print(f"log-log slope N* vs A: measured {slope_m:.3f}, predicted {slope_p:.3f}  ('*' = predicted detectable)")
RESULTS["C"] = dict(C1=rowsC1, C1_pred_opt=best[0], C2=rowsC2, slope_meas=float(slope_m), slope_pred=float(slope_p))
print(f"Part C done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part D — attacker design zoo, raw vs intensity-whitened statistic
# **S1** benign uniform · **S2** benign *non-uniform* (50% uniform + 50% band in y). Reference intensity estimated from a historical benign sample.
#
# Per design we report, for RAW and WHITENED pipelines: `clean_min` = exactly one eigenvalue below the λ_min threshold (a genuine missing mode), `multi_min` = ≥2 below (bulk deformation, *not* attributable), `max` = ≥1 above the λ_max threshold (distribution shift / concentration).
#
# | Scenario | Design | Locked prediction (whitened) |
# |---|---|---|
# | S1 | jittered, sobol, k-center(FPS) | clean_min ≈ 1, max ≈ 0 |
# | S1 | random_from_pool | all ≈ 0 |
# | S1 | uncertainty_band, kcenter_in_band | max ≈ 1; clean_min ≈ 0 |
# | S2 | natural_iid (= null) | RAW fires falsely; WHITENED all ≈ 0 |
# | S2 | **warped_jitter** (same marginal as benign, stratified) | **clean_min ≈ 1, max ≈ 0** — detectable although its query distribution is identical to benign |
# | S2 | kcenter_from_benign_pool, uncertainty_band | max ≈ 1; clean_min exploratory |

# %%
t0 = time.time()
N0 = BASE["L"] ** 2; aD = 0.005
tmin, tmax = thr_min(BASE["A"], m0, aD), thr_max(BASE["A"], m0, aD)
def kcenter_benign_pool(N, rng): return _fps(benign_bump(10 * N, rng), N, rng)
def uncertainty_benign_pool(N, rng):
    pool = benign_bump(10 * N, rng); dist = np.abs(pool[:, 1] - boundary(pool[:, 0])); return pool[np.argsort(dist)[:N]]
ZOO = [
    ("S1", "jittered_grid",       lambda rg: d_jitter(BASE["L"], rg),            benign_uniform, REF_UNIF),
    ("S1", "sobol_scrambled",     lambda rg: d_sobol(N0, rg),                    benign_uniform, REF_UNIF),
    ("S1", "kcenter_FPS_pool",    lambda rg: d_kcenter(N0, rg),                  benign_uniform, REF_UNIF),
    ("S1", "random_from_pool",    lambda rg: d_random_pool(N0, rg),              benign_uniform, REF_UNIF),
    ("S1", "uncertainty_band",    lambda rg: d_uncertainty(N0, rg),              benign_uniform, REF_UNIF),
    ("S1", "kcenter_in_band",     lambda rg: d_band_kcenter(N0, rg),             benign_uniform, REF_UNIF),
    ("S2", "natural_iid(=null)",  lambda rg: benign_bump(N0, rg),                benign_bump,    REF_BUMP),
    ("S2", "warped_jitter",       lambda rg: warp_to_bump(d_jitter(BASE["L"], rg)), benign_bump, REF_BUMP),
    ("S2", "kcenter_benign_pool", lambda rg: kcenter_benign_pool(N0, rg),        benign_bump,    REF_BUMP),
    ("S2", "uncertainty_band",    lambda rg: uncertainty_benign_pool(N0, rg),    benign_bump,    REF_BUMP),
]
rowsD = []
print(f"thresholds (per side alpha={aD}): lam_min < {tmin:.4f} | lam_max > {tmax:.4f}")
hdr = f"{'sc':>2} {'design':>20} {'S_un_w':>7} | {'RAW cln':>7} {'mult':>5} {'max':>5} | {'WHT cln':>7} {'mult':>5} {'max':>5} {'ov_min':>6} {'ov_max':>6}"
print(hdr)
for i, (sc, name, fn, bfn, ref) in enumerate(ZOO):
    acc = []
    for t in range(TR_D):
        rng = np.random.default_rng(SEED + 50_000 + 1000 * i + t)
        pts, lab, cnt = build_world(fn(rng), BASE["K"], BASE["A"], 0.0, rng, benign_fn=bfn)
        acc.append(spectra_raw_white(pts, lab, cnt, modes0, BASE["K"], ref, tmin, tmax))
    row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}; row.update(scenario=sc, design=name)
    rowsD.append(row)
    # [PAPER] Prediction log (Step 2): S2 warped_jitter (same-marginal stratified attacker), whitened clean λ_min 1.00
    print(f"{sc:>2} {name:>20} {row['S_union_wht']:7.3f} | {row['raw_clean_min']:7.2f} {row['raw_multi_min']:5.2f} {row['raw_max']:5.2f} | "
          f"{row['wht_clean_min']:7.2f} {row['wht_multi_min']:5.2f} {row['wht_max']:5.2f} {row['wht_ov_min']:6.2f} {row['wht_ov_max']:6.2f}")
RESULTS["D"] = dict(thr_min=tmin, thr_max=tmax, rows=rowsD)
print(f"Part D done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
# A: null lam_min vs TW density
z = (mu_m - NULL_LMIN) / sg_m
ax[0,0].hist(z, bins=80, density=True, alpha=.6, label=f"null (n={len(z)}) rescaled")
ax[0,0].plot(TW_GRID, TW_PDF, "k-", label="Tracy–Widom β=2")
ax[0,0].set_xlabel("(μ₋ − λ_min)/σ₋"); ax[0,0].set_title("A: analytic null calibration"); ax[0,0].legend()
# B: predicted vs measured f50
Bm = [c for c in RESULTS["B"] if c["f50_meas"] == c["f50_meas"]]
for c in Bm: ax[0,1].scatter(c["f50_pred"], c["f50_meas"], s=20 + c["L"] / 2, label=f"L={c['L']}, γ={c['gamma']:.2f}")
lim = [0, max([c["f50_pred"] for c in RESULTS["B"]] + [0.5]) + 0.05]
ax[0,1].plot(lim, lim, "k:"); ax[0,1].set_xlabel("predicted f50 (TW)"); ax[0,1].set_ylabel("measured f50")
ax[0,1].set_title("B: prospective threshold test"); ax[0,1].legend(fontsize=7)
# B: overlap pred vs meas
for c in RESULTS["B"]: ax[0,2].scatter(c["ov_pred"], c["ov_meas"], s=12)
ax[0,2].plot([0, 1], [0, 1], "k:"); ax[0,2].set_xlabel("predicted overlap"); ax[0,2].set_ylabel("measured overlap")
ax[0,2].set_title("B: eigenvector overlap formula")
# C1
if RESULTS["C"]["C1"]:
    ax[1,0].plot([r["nmax"] for r in RESULTS["C"]["C1"]], [r["f50_pred"] for r in RESULTS["C"]["C1"]], "s--", label="predicted")
    ax[1,0].plot([r["nmax"] for r in RESULTS["C"]["C1"]], [r["f50_meas"] for r in RESULTS["C"]["C1"]], "o-", label="measured")
ax[1,0].set_xlabel("window n_max"); ax[1,0].set_ylabel("attacker hiding budget f50"); ax[1,0].set_title("C1: defender's optimal window"); ax[1,0].legend()
# C2
rc = RESULTS["C"]["C2"]
ax[1,1].loglog([r["A"] for r in rc], [r["Nstar_pred"] for r in rc], "s--", label=f"predicted (slope {RESULTS['C']['slope_pred']:.2f})")
ax[1,1].loglog([r["A"] for r in rc], [r["Nstar_meas"] for r in rc], "o-", label=f"measured (slope {RESULTS['C']['slope_meas']:.2f})")
ax[1,1].set_xlabel("monitored accounts A"); ax[1,1].set_ylabel("critical campaign size N*"); ax[1,1].set_title("C2: scaling law"); ax[1,1].legend()
# D
rd = RESULTS["D"]["rows"]; names = [f"{r['scenario']}:{r['design']}" for r in rd]; xs = np.arange(len(names))
ax[1,2].bar(xs - 0.27, [r["wht_clean_min"] for r in rd], 0.27, label="whitened: clean λ_min")
ax[1,2].bar(xs, [r["wht_max"] for r in rd], 0.27, label="whitened: λ_max")
ax[1,2].bar(xs + 0.27, [r["raw_clean_min"] + r["raw_multi_min"] for r in rd], 0.27, alpha=.5, label="raw: any λ_min")
ax[1,2].set_xticks(xs); ax[1,2].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax[1,2].set_ylabel("rate"); ax[1,2].set_title("D: design zoo"); ax[1,2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step2_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy this cell's output into the chat

# %%
R = RESULTS
print("=" * 78); print("STEP 2 VERDICT REPORT"); print("=" * 78)
print("config:", json.dumps(R["config"]), "| gpu_check:", R.get("gpu_check", "n/a"))
a = R["A"]
print(f"[A] null lam_min {a['lmin_mean']:.5f}±{a['lmin_sd']:.5f} vs TW {a['lmin_pred']:.5f}±{a['lmin_sd_pred']:.5f}; lam_max {a['lmax_mean']:.4f} vs TW {a['lmax_pred']:.4f}")
print("[A] FPR (min / max) at alpha: " + "; ".join(f"{r['alpha']}: {r['fpr_min']:.5f}({r['cnt_min']}) / {r['fpr_max']:.5f}({r['cnt_max']})" for r in a["rows"]))
print(f"[A3] non-uniform null @0.005/side: RAW any-min={a['A3']['raw_clean_min']+a['A3']['raw_multi_min']:.4f} max={a['A3']['raw_max']:.4f} | WHITENED any-min={a['A3']['wht_clean_min']+a['A3']['wht_multi_min']:.4f} max={a['A3']['wht_max']:.4f}")
print("[A2] hetero sizes FPR (min/max): " + "; ".join(f"{r['alpha']}: {r['fpr_min']:.4f}/{r['fpr_max']:.4f}" for r in a["hetero"]))
for c in R["B"]:
    print(f"[B] L={c['L']:3d} m={c['m']:5d} A={c['A']:4d} g={c['gamma']:.2f}: f50 meas={c['f50_meas']:.3f} TWpred={c['f50_pred']:.3f} asym={c['fstar']:.3f} FPRchk={c['fpr_check']:.3f}")
sup = [(p, mv) for c in R["B"] for p, mv in zip(c["ov_pred"], c["ov_meas"]) if p > 0.5]
if sup: print(f"[B] overlap formula (pred>0.5, n={len(sup)}): mean |pred-meas| = {np.mean([abs(p-mv) for p, mv in sup]):.4f}")
print("[C1] nmax: f50 pred/meas -> " + "; ".join(f"{r['nmax']}: {r['f50_pred']:.3f}/{r['f50_meas']:.3f}" for r in R["C"]["C1"]) + f" | predicted optimum nmax={R['C']['C1_pred_opt']}")
print("[C2] " + "; ".join(f"A={r['A']}: N* meas={r['Nstar_meas']:.0f} pred={r['Nstar_pred']:.0f}" for r in R["C"]["C2"]))
print(f"[C2] log-log slope measured={R['C']['slope_meas']:.3f} predicted={R['C']['slope_pred']:.3f}")
for r in R["D"]["rows"]:
    print(f"[D] {r['scenario']} {r['design']:>20}: RAW clean/multi/max={r['raw_clean_min']:.2f}/{r['raw_multi_min']:.2f}/{r['raw_max']:.2f} | "
          f"WHT clean/multi/max={r['wht_clean_min']:.2f}/{r['wht_multi_min']:.2f}/{r['wht_max']:.2f} ov_min={r['wht_ov_min']:.2f} ov_max={r['wht_ov_max']:.2f} S_un_w={r['S_union_wht']:.3f}")
print("=" * 78)
with open(f"{OUTDIR}/step2_results.json", "w") as fh: json.dump(R, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step2_results.json and step2_figure.png")
