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
# | Prediction log (Step 3), row "High-dimensional analytic calibration" (P-A) | analytic TW false alarms in d = 2/8/32; fixed by self-normalised rows and simulation-calibrated thresholds | Fail, then fixed (consequence: row normalisation) | config cell `ROWNORM = True`; `evaluate(...)` row normalisation; Part A cell, print "analytic-TW FPR" and empirical thresholds `emp_tmin`/`emp_tmax` |
# | Prediction log (Step 3), row "Herding detectable in every dimension" (P-B2) | whitened union S̄ of `herding` campaigns | Fail (S̄ 0.70–0.89) | Part B cell (S per bandwidth) and Part C summary, column S_union, design `herding`; VERDICT REPORT "[B]"/"[C]" lines |
# | Prediction log (Step 3), row "Mismatched herding detectable" (P-B4) | whitened union S̄ of `herding_mismatched` | Fail (S̄ ≈ 1) | same cells, design `herding_mismatched` |
# | Prediction log (Step 3), row "k-means++ shows λmax" (P-B3) | λ_max rate of `kmeanspp` | Fail (looks i.i.d.) | same cells, design `kmeanspp`, column max |
#
# The herding failure led to the detectability law tested in Step 4 (Prediction log, Step 3 consequence "Led to the law").
#
# Not reported in the paper: `iid`, `class_balanced` and `kcenter_FPS` rates, oracle (known-colluder) tests,
# spectral vs pairwise linkage AUCs, the BBP-transfer check (P-B5), the whitening self-check, Gamma-oracle FPR,
# step3_figure.png and step3_results.json.

# %% [markdown]
# # Step 3 — High-dimensional generalization: random Fourier modes, realistic selection strategies, baselines
# **Kaggle:** GPU accelerator → **Run All** (≈25–45 min). `QUICK=True` for a smoke test.
#
# **Why.** Real query embeddings live in $\mathbb R^d$ with $d\gg2$. Lattice wavevectors no longer exist, so we use **random Fourier modes** $\omega_j\sim\mathcal N(0,h^{-2}I_d)$; the bandwidth $h$ plays the role of $n_{\max}$. Each account's whitened vector is **self-normalised** (unit RMS): in high dimension single queries carry very unequal leverage after whitening, which otherwise spreads both spectral edges (found in the smoke test). Benign traffic is an anisotropic Gaussian mixture (power-law spectrum, effective dimension ≈ 2 / 6 / 24 for $d$ = 2 / 8 / 32) whose characteristic function is known **exactly**, so whitening $\Sigma(\omega,\omega')=\varphi(\omega-\omega')-\varphi(\omega)\overline{\varphi(\omega')}$ is exact (reference-size effects were quantified in Step 2). Attackers select a campaign of $N$ queries from a pool of **natural** (benign-distributed) data — the strong, in-distribution adversary.
#
# **Attacker strategies.** `iid` (null) · `class_balanced` (equal per predicted class) · `kmeans++` (D² seeding) · `kcenter_FPS` (ActiveThief-style) · `herding` (MMD-optimal coreset in the defender's geometry) · `herding_mismatched` (herding in the attacker's *own* feature space $g(x)$, a random MLP).
#
# **Baselines.** *Oracle* tests that are told who the colluders are (union fluctuation too small / too large), and a *pairwise-similarity linkage* test (FDINet-style logic: colluders look alike).
#
# | ID | Locked prediction |
# |---|---|
# | **P-A** | *(revised after smoke test)* Rows are self-normalised. Analytic TW thresholds hold at moderate bandwidth; at the lowest frequencies (ill-conditioned Σ) they may drift, so Part B uses **simulation-calibrated** thresholds from Part A (a defender can do the same by resampling historical accounts) |
# | **P-B1** | `iid`: nothing fires. `class_balanced`: weak at most (clean-λ_min power ≤ 0.2) |
# | **P-B2** | `herding`: clean λ_min outlier at low frequency (large $h$) in **every** $d$ |
# | **P-B3** | `kcenter_FPS`, `kmeans++`: missing mode weakens as effective dimension grows; in $d$=32, λ_max (tail over-sampling) dominates |
# | **P-B4** | `herding_mismatched`: weaker than matched herding, but detectable at the lowest frequencies |
# | **P-B5** | BBP law transfers: measured λ_min ≈ $(1+\ell)(1+\gamma/\ell)$ with $\ell=\bar S_{\rm union}-1$ when supercritical |
# | **P-C** | Spectral linkage AUC ≫ pairwise-similarity AUC (≈0.5) wherever the missing mode fires |
#
# Copy the **VERDICT REPORT** and the figure back into the chat.

# %%
# ===== Cell 1: configuration & backend =====
import math, json, time, os
import numpy as np
from scipy.special import airy
from scipy.optimize import brentq
from scipy.stats import rankdata, gamma as gamma_dist

QUICK = False
SEED = 3031
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
CHUNK = 65536 if USE_GPU else 8192

A, K, M_MODES, NCOMP = 100, 20, 400, 10
if QUICK:
    DIMS, HS, TR, N_NULL, N_CAMP = [2, 8], [0.5, 4.0], 3, 60, 600
elif USE_GPU:
    DIMS, HS, TR, N_NULL, N_CAMP = [2, 8, 32], [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], 40, 2000, 1500
else:
    DIMS, HS, TR, N_NULL, N_CAMP = [2, 8, 32], [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], 12, 400, 1500
# [PAPER] Prediction log (Step 3): "High-dimensional analytic calibration" Fail, then fixed by row normalisation (ROWNORM)
POOL_MULT, ALPHA_SIDE, ROWNORM = 6, 0.005, True
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU/numpy'} | QUICK={QUICK}")
print(f"DIMS={DIMS} HS={HS} TR={TR} N_NULL={N_NULL} N_CAMP={N_CAMP} A={A} K={K} m={M_MODES}")


# %%
# ===== Cell 2: Tracy-Widom beta=2 and random-matrix helpers (same as Step 2) =====
def tw2_cdf(s, n=60, Lw=14.0):
    x, w = np.polynomial.legendre.leggauss(n); x = s + (x + 1) * Lw / 2; w = w * Lw / 2
    ai, aip, _, _ = airy(x); X, Y = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        Kk = (ai[:, None] * aip[None, :] - aip[:, None] * ai[None, :]) / (X - Y)
    Kk[np.diag_indices(n)] = aip ** 2 - x * ai ** 2; sw = np.sqrt(w)
    return float(np.linalg.det(np.eye(n) - sw[:, None] * Kk * sw[None, :]))
_TWQ = {}
def tw_q(p):
    if p not in _TWQ: _TWQ[p] = brentq(lambda v: tw2_cdf(v) - p, -9, 8)
    return _TWQ[p]
TW_MEAN = -1.7711
def edges(p, n):
    a, b = math.sqrt(n), math.sqrt(p)
    return ((a - b) ** 2 / n, (a - b) * (1 / b - 1 / a) ** (1 / 3) / n, (a + b) ** 2 / n, (a + b) * (1 / b + 1 / a) ** (1 / 3) / n)
def thr_min(p, n, alpha): mu, sg, _, _ = edges(p, n); return mu - sg * tw_q(1 - alpha)
def thr_max(p, n, alpha): _, _, mu, sg = edges(p, n); return mu + sg * tw_q(1 - alpha)
def bbp_low(l, g): return (1 + l) * (1 + g / l)
print(f"TW q99={tw_q(0.99):.4f} (ref 0.4776)")


# %%
# ===== Cell 3: benign model, exact whitening, Fourier, attacker designs =====
def _fourier_numpy(pts, labels, nacc, modes):
    import scipy.sparse as sp
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
def fourier_accounts(pts, labels, nacc, modes):   # rho_a(w) = sum_j exp(-2 pi i modes.x_j)
    return _fourier_torch(pts, labels, nacc, modes) if USE_GPU else _fourier_numpy(pts, labels, nacc, modes)

class GMM:
    # anisotropic Gaussian mixture, power-law spectrum lam_i ~ i^-0.5, trace 4; 60% between / 40% within
    def __init__(self, d, seed):
        r = np.random.default_rng(seed); lam = np.arange(1, d + 1) ** -0.5; lam = 4 * lam / lam.sum()
        self.d, self.lw = d, 0.4 * lam
        self.mu = r.standard_normal((NCOMP, d)) * np.sqrt(0.6 * lam)
        self.eff_dim = lam.sum() ** 2 / (lam ** 2).sum()
    def sample(self, n, rng, return_c=False):
        c = rng.integers(0, NCOMP, n); x = self.mu[c] + rng.standard_normal((n, self.d)) * np.sqrt(self.lw)
        return (x, c) if return_c else x
    def cf(self, W):   # phi(w) = E exp(-i w.x), W (..., d)
        return np.exp(-1j * (W @ self.mu.T)).mean(-1) * np.exp(-0.5 * (W ** 2 @ self.lw))

def make_bank(gmm, h, seed, floor=1e-5):
    # random Fourier modes + exact null covariance -> whitening projector (rank m_eff)
    r = np.random.default_rng(seed); Om = r.standard_normal((M_MODES, gmm.d)) / h
    mu = gmm.cf(Om)
    Sig = gmm.cf(Om[:, None, :] - Om[None, :, :]) - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew, U = np.linalg.eigh(Sig); keep = ew > floor * ew.max()
    Wk = U[:, keep].conj() / np.sqrt(ew[keep])          # row-form whitening: F_w = F_c @ Wk
    me = int(keep.sum()); ok = A / me <= 0.8
    return dict(h=h, modes=Om / (2 * np.pi), mu=mu, Wk=Wk, meff=me, ok=ok, cond=float(ew.max() / max(ew[keep].min(), 1e-300)),
                tmin=thr_min(A, me, ALPHA_SIDE) if ok else np.nan, tmax=thr_max(A, me, ALPHA_SIDE) if ok else np.nan)

# ---------------- attacker designs: choose N points from a natural pool (pool ~ benign) ----------------
def _sqd(P, x): d = P - x; return (d * d).sum(1)
def sel_iid(pool, c, N, rng): return pool[rng.choice(len(pool), N, replace=False)]
def sel_class_balanced(pool, c, N, rng):
    per = N // NCOMP; idx = np.concatenate([rng.choice(np.where(c == k)[0], per, replace=False) for k in range(NCOMP)])
    extra = rng.choice(np.setdiff1d(np.arange(len(pool)), idx), N - len(idx), replace=False)
    return pool[np.concatenate([idx, extra])]
def sel_fps(pool, c, N, rng):
    sel = [int(rng.integers(len(pool)))]; d2 = _sqd(pool, pool[sel[0]])
    for _ in range(N - 1):
        i = int(np.argmax(d2)); sel.append(i); d2 = np.minimum(d2, _sqd(pool, pool[i]))
    return pool[np.array(sel)]
def sel_kmeanspp(pool, c, N, rng):
    sel = [int(rng.integers(len(pool)))]; d2 = _sqd(pool, pool[sel[0]])
    for _ in range(N - 1):
        p = d2 / d2.sum(); i = int(rng.choice(len(pool), p=p)); sel.append(i); d2 = np.minimum(d2, _sqd(pool, pool[i]))
    return pool[np.array(sel)]
def _herd(feat_space, N, rng, h_att=1.0, D=256):
    # kernel herding (greedy MMD minimisation) with random Fourier features of a Gaussian kernel
    d = feat_space.shape[1]; W = rng.standard_normal((d, D)) / h_att; b = rng.uniform(0, 2 * np.pi, D)
    Phi = np.sqrt(2.0 / D) * np.cos(feat_space @ W + b); target = Phi.mean(0)
    s = np.zeros(D); score_t = Phi @ target; taken = np.zeros(len(Phi), bool); sel = []
    for t in range(N):
        sc = score_t - (Phi @ s) / (t + 1); sc[taken] = -np.inf
        i = int(np.argmax(sc)); sel.append(i); taken[i] = True; s += Phi[i]
    return np.array(sel)
def sel_herding(pool, c, N, rng): return pool[_herd(pool, N, rng)]
_GNET = {}
def g_map(X, d):   # attacker's own representation: fixed random 2-layer MLP, standardized
    if d not in _GNET:
        r = np.random.default_rng(777 + d)
        _GNET[d] = (r.standard_normal((d, 2 * d)) * 1.5 / np.sqrt(d), r.standard_normal((2 * d, d)) / np.sqrt(2 * d))
    W1, W2 = _GNET[d]; Z = np.tanh(X @ W1) @ W2; return (Z - Z.mean(0)) / Z.std(0)
def sel_herding_mismatched(pool, c, N, rng): return pool[_herd(g_map(pool, pool.shape[1]), N, rng)]
DESIGNS = dict(iid=sel_iid, class_balanced=sel_class_balanced, kmeanspp=sel_kmeanspp, kcenter_FPS=sel_fps,
               herding=sel_herding, herding_mismatched=sel_herding_mismatched)

def build_world(gmm, design, rng):
    pool, c = gmm.sample(POOL_MULT * N_CAMP, rng, return_c=True)
    camp = DESIGNS[design](pool, c, N_CAMP, rng); camp = camp[rng.permutation(len(camp))]
    lab_c = rng.integers(0, K, len(camp)); n_per = len(camp) // K
    ben = gmm.sample((A - K) * n_per, rng); lab_b = K + np.repeat(np.arange(A - K), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b])
    return pts, lab, np.bincount(lab, minlength=A)

def _auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))
IU = np.triu_indices(A, 1); PAIR_COL = (IU[0] < K) & (IU[1] < K)

def evaluate(pts, lab, cnt, banks, linkage=True):
    allmodes = np.vstack([b["modes"] for b in banks]); Fall = fourier_accounts(pts, lab, A, allmodes)
    sq = np.sqrt(cnt); Ncol = cnt[:K].sum(); out = []
    for j, b in enumerate(banks):
        if not b["ok"]: out.append(None); continue
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]
        Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        S_un = float((np.abs((Fw[:K] * sq[:K, None]).sum(0)) ** 2).mean() / Ncol)   # oracle union statistic
        if ROWNORM: Fw = Fw / np.sqrt((np.abs(Fw) ** 2).mean(1))[:, None]          # self-normalised rows
        ev, V = np.linalg.eigh(Fw @ Fw.conj().T / b["meff"])
        tmn, tmx = b.get("emp_tmin", b["tmin"]), b.get("emp_tmax", b["tmax"])
        nb_, na_ = int((ev < tmn).sum()), int((ev > tmx).sum())
        g = A / b["meff"]; l = S_un - 1
        r = dict(clean_min=float(nb_ == 1), multi_min=float(nb_ >= 2), max=float(na_ >= 1), lam_min=ev[0], lam_max=ev[-1],
                 S_union=S_un, lam_bbp=bbp_low(l, g) if l < -math.sqrt(g) else edges(A, b["meff"])[0],
                 oracle_low=float(S_un < b.get("emp_olow", gamma_dist.ppf(0.01, b["meff"]) / b["meff"])),
                 oracle_high=float(S_un > b.get("emp_ohigh", gamma_dist.ppf(0.99, b["meff"]) / b["meff"])))
        if linkage:
            r["auc_spectral"] = _auc(np.abs(V[:K, 0]) ** 2, np.abs(V[K:, 0]) ** 2)
            mu_a = Fr / cnt[:, None]; G = mu_a @ mu_a.conj().T; nrm = np.real(np.diag(G))
            D2 = (nrm[:, None] + nrm[None, :] - 2 * np.real(G))[IU]
            r["auc_pairwise"] = _auc(-D2[PAIR_COL], -D2[~PAIR_COL])
        out.append(r)
    return out

GMMS = {d: GMM(d, 100 + d) for d in DIMS}
BANKS = {d: [make_bank(GMMS[d], h, 5000 + 10 * d + int(10 * h)) for h in HS] for d in DIMS}
print(f"{'d':>3} {'eff_dim':>7} | " + " ".join(f"h={h}: m_eff/cond" for h in HS))
for d in DIMS:
    print(f"{d:3d} {GMMS[d].eff_dim:7.2f} | " + "  ".join(f"{b['meff']:4d}/{b['cond']:.0e}{'' if b['ok'] else '(skip)'}" for b in BANKS[d]))
RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, DIMS=DIMS, HS=HS, TR=TR, N_NULL=N_NULL, N_CAMP=N_CAMP, A=A, K=K, m=M_MODES,
                          backend="gpu" if USE_GPU else "cpu"),
           "banks": {str(d): [dict(h=b["h"], meff=b["meff"], ok=b["ok"], cond=b["cond"]) for b in BANKS[d]] for d in DIMS},
           "eff_dim": {str(d): float(GMMS[d].eff_dim) for d in DIMS}}
# self-check: whitening exactness on a large iid sample (whitened mode variance should be ~1)
rng = np.random.default_rng(1); X = GMMS[DIMS[-1]].sample(200000, rng); b = BANKS[DIMS[-1]][1]
Fc = (np.exp(-2j * np.pi * (X[:20000] @ b["modes"].T)) - b["mu"]) @ b["Wk"]
print(f"whitening check (d={DIMS[-1]}, h={b['h']}): mean |z|^2 = {np.mean(np.abs(Fc)**2):.3f} (expect 1.000)")

# %% [markdown]
# ## Part A — null calibration in high dimension (iid benign everywhere)

# %%
t0 = time.time(); rowsA = []
for d in DIMS:
    acc = {j: [] for j in range(len(HS))}
    for t in range(N_NULL):
        rng = np.random.default_rng(SEED + 1_000_000 * d + t)
        pts, lab, cnt = build_world(GMMS[d], "iid", rng)
        for j, r in enumerate(evaluate(pts, lab, cnt, BANKS[d], linkage=False)):
            if r is not None: acc[j].append(r)
    for j, b in enumerate(BANKS[d]):
        if not b["ok"]:
            rowsA.append(dict(d=d, h=b["h"], ok=False)); print(f"d={d:2d} h={b['h']:4.1f}: untestable (m_eff={b['meff']})"); continue
        lmn = np.array([r["lam_min"] for r in acc[j]]); lmx = np.array([r["lam_max"] for r in acc[j]])
        Sn = np.array([r["S_union"] for r in acc[j]])
        mu_m, sg_m, mu_p, sg_p = edges(A, b["meff"])
        # empirical (simulation-calibrated) thresholds used from now on in Part B
        b["emp_tmin"], b["emp_tmax"] = float(np.quantile(lmn, ALPHA_SIDE)), float(np.quantile(lmx, 1 - ALPHA_SIDE))
        b["emp_olow"], b["emp_ohigh"] = float(np.quantile(Sn, 0.01)), float(np.quantile(Sn, 0.99))
        row = dict(d=d, h=b["h"], ok=True, meff=b["meff"], n=len(lmn),
                   fpr_min_TW=float(np.mean(lmn < b["tmin"])), fpr_max_TW=float(np.mean(lmx > b["tmax"])),
                   z_min=float((lmn.mean() - (mu_m - sg_m * TW_MEAN)) / sg_m), z_max=float((lmx.mean() - (mu_p + sg_p * TW_MEAN)) / sg_p),
                   tmin_TW=b["tmin"], tmin_emp=b["emp_tmin"], tmax_TW=b["tmax"], tmax_emp=b["emp_tmax"],
                   oracle_low_gamma=float(np.mean(Sn < gamma_dist.ppf(0.01, b["meff"]) / b["meff"])),
                   oracle_high_gamma=float(np.mean(Sn > gamma_dist.ppf(0.99, b["meff"]) / b["meff"])))
        rowsA.append(row)
        # [PAPER] Prediction log (Step 3): analytic TW calibration in high dimension (Fail, then fixed); empirical thresholds above are used from here on
        print(f"d={d:2d} h={b['h']:4.2f} m_eff={b['meff']:3d}: analytic-TW FPR min={row['fpr_min_TW']:.4f} max={row['fpr_max_TW']:.4f} (nom {ALPHA_SIDE}) "
              f"z={row['z_min']:+.2f}/{row['z_max']:+.2f} | thr TW/emp min {b['tmin']:.4f}/{b['emp_tmin']:.4f} max {b['tmax']:.4f}/{b['emp_tmax']:.4f} "
              f"| Gamma-oracle FPR {row['oracle_low_gamma']:.3f}/{row['oracle_high_gamma']:.3f}")
RESULTS["A"] = rowsA; print(f"Part A done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part B — detection by attacker strategy × dimension × bandwidth

# %%
t0 = time.time(); RAW = {}
for d in DIMS:
    for name in DESIGNS:
        t1 = time.time(); acc = {j: [] for j in range(len(HS))}
        for t in range(TR):
            rng = np.random.default_rng(SEED + 7_000_000 + 100_000 * d + 1000 * list(DESIGNS).index(name) + t)
            pts, lab, cnt = build_world(GMMS[d], name, rng)
            for j, r in enumerate(evaluate(pts, lab, cnt, BANKS[d])):
                if r is not None: acc[j].append(r)
        RAW[(d, name)] = acc
        line = []
        for j, b in enumerate(BANKS[d]):
            if not acc[j]: line.append(f"h{b['h']}: --"); continue
            m_ = lambda k: np.mean([r[k] for r in acc[j]])
            line.append(f"h{b['h']}: cln {m_('clean_min'):.2f} max {m_('max'):.2f} S {m_('S_union'):.3f}")
        print(f"d={d:2d} {name:>19} ({time.time()-t1:4.0f}s) | " + " | ".join(line))
rowsB = []
for (d, name), acc in RAW.items():
    for j, b in enumerate(BANKS[d]):
        if not acc[j]: continue
        m_ = lambda k: float(np.mean([r[k] for r in acc[j]]))
        rowsB.append(dict(d=d, design=name, h=b["h"], meff=b["meff"], clean_min=m_("clean_min"), multi_min=m_("multi_min"),
                          max=m_("max"), S_union=m_("S_union"), lam_min=m_("lam_min"), lam_bbp=m_("lam_bbp"),
                          oracle_low=m_("oracle_low"), oracle_high=m_("oracle_high"),
                          auc_spectral=m_("auc_spectral"), auc_pairwise=m_("auc_pairwise")))
RESULTS["B"] = rowsB; print(f"Part B done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Part C — best-bandwidth summary, baselines and linkage

# %%
print(f"{'d':>3} {'design':>19} {'h*':>4} | {'clean_min':>9} {'max':>5} {'multi':>5} | {'oracleLow':>9} {'oracleHigh':>10} | "
      f"{'AUC_spec':>8} {'AUC_pair':>8} | {'lam_min':>7} {'lam_BBP':>7}")
summary = []
for d in DIMS:
    for name in DESIGNS:
        rs = [r for r in rowsB if r["d"] == d and r["design"] == name]
        if not rs: continue
        best = max(rs, key=lambda r: (r["clean_min"], -r["S_union"]))   # defender's best bandwidth for the missing mode
        summary.append(best)
        # [PAPER] Prediction log (Step 3): herding S̄ 0.70–0.89 (Fail); herding_mismatched S̄ ≈ 1 (Fail); kmeanspp λ_max "looks i.i.d." (Fail)
        print(f"{d:3d} {name:>19} {best['h']:4.1f} | {best['clean_min']:9.2f} {best['max']:5.2f} {best['multi_min']:5.2f} | "
              f"{best['oracle_low']:9.2f} {best['oracle_high']:10.2f} | {best['auc_spectral']:8.3f} {best['auc_pairwise']:8.3f} | "
              f"{best['lam_min']:7.4f} {best['lam_bbp']:7.4f}")
RESULTS["C"] = summary
sup = [r for r in rowsB if r["clean_min"] > 0.9 and r["lam_bbp"] < edges(A, r["meff"])[0] - 0.02]
if sup:
    err = np.array([(r["lam_min"] - r["lam_bbp"]) / max(r["lam_bbp"], 1e-9) for r in sup])
    print(f"\nP-B5 BBP transfer (supercritical configs n={len(sup)}): median rel.err {np.median(err):+.3f}, max |rel.err| {np.abs(err).max():.3f}")
    RESULTS["bbp_transfer"] = dict(n=len(sup), median=float(np.median(err)), maxabs=float(np.abs(err).max()))

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
names = list(DESIGNS); cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for i, d in enumerate(DIMS[:3]):
    a = ax[0, i]
    for k, name in enumerate(names):
        rs = sorted([r for r in rowsB if r["d"] == d and r["design"] == name], key=lambda r: r["h"])
        a.plot([r["h"] for r in rs], [r["clean_min"] for r in rs], "o-", color=cols[k], label=name)
        a.plot([r["h"] for r in rs], [r["max"] for r in rs], "x:", color=cols[k], alpha=.6)
    a.set_xscale("log"); a.set_ylim(-.05, 1.05); a.set_xlabel("bandwidth h (low frequency →)")
    a.set_title(f"d={d} (eff.dim {GMMS[d].eff_dim:.1f}): clean λ_min (—o) / λ_max (:x)")
    if i == 0: a.legend(fontsize=7)
a = ax[1, 0]
for k, name in enumerate(names):
    rs = [r for r in RESULTS["C"] if r["design"] == name]
    a.plot([GMMS[r["d"]].eff_dim for r in rs], [r["S_union"] for r in rs], "o-", color=cols[k], label=name)
a.axhline(1, color="k", ls=":"); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("effective dimension")
a.set_ylabel("whitened union S̄ at best h"); a.set_title("Suppression vs dimension (S̄<1 = sub-Poisson)"); a.legend(fontsize=7)
a = ax[1, 1]; rs = RESULTS["C"]; xs = np.arange(len(rs))
a.bar(xs - 0.2, [r["auc_spectral"] for r in rs], 0.4, label="spectral (no linkage needed)")
a.bar(xs + 0.2, [r["auc_pairwise"] for r in rs], 0.4, label="pairwise similarity")
a.axhline(0.5, color="k", ls=":"); a.set_xticks(xs); a.set_xticklabels([f"d{r['d']}:{r['design']}" for r in rs], rotation=60, ha="right", fontsize=6)
a.set_ylabel("colluder identification AUC"); a.set_title("Linkage: who are the colluders?"); a.legend(fontsize=7)
a = ax[1, 2]; sp_ = [r for r in rowsB if r["clean_min"] > 0.5]
a.scatter([r["lam_bbp"] for r in sp_], [r["lam_min"] for r in sp_], c=[DIMS.index(r["d"]) for r in sp_], cmap="viridis", s=18)
lim = [0, max([r["lam_min"] for r in sp_] + [0.3])]; a.plot(lim, lim, "k:")
a.set_xlabel("BBP prediction from measured S̄"); a.set_ylabel("measured λ_min"); a.set_title("P-B5: BBP law in high dimension")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step3_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy this cell's output into the chat

# %%
R = RESULTS
print("=" * 80); print("STEP 3 VERDICT REPORT"); print("=" * 80)
print("config:", json.dumps(R["config"])); print("eff_dim:", {k: round(v, 2) for k, v in R["eff_dim"].items()})
print("banks (h: m_eff):", {d: [(b["h"], b["meff"]) for b in v] for d, v in R["banks"].items()})
for r in R["A"]:
    if r["ok"]:
        print(f"[A] d={r['d']:2d} h={r['h']:4.2f} m_eff={r['meff']}: TW-FPR min/max={r['fpr_min_TW']:.4f}/{r['fpr_max_TW']:.4f} (nom {ALPHA_SIDE}) z={r['z_min']:+.2f}/{r['z_max']:+.2f} thr_min TW/emp={r['tmin_TW']:.4f}/{r['tmin_emp']:.4f} GammaOracle={r['oracle_low_gamma']:.3f}/{r['oracle_high_gamma']:.3f}")
for d in DIMS:
    for name in DESIGNS:
        rs = sorted([r for r in R["B"] if r["d"] == d and r["design"] == name], key=lambda r: r["h"])
        print(f"[B] d={d:2d} {name:>19}: " + " ".join(f"h{r['h']}:{r['clean_min']:.2f}/{r['max']:.2f}/S{r['S_union']:.2f}" for r in rs))
for r in R["C"]:
    print(f"[C] d={r['d']:2d} {r['design']:>19} h*={r['h']}: clean={r['clean_min']:.2f} max={r['max']:.2f} multi={r['multi_min']:.2f} "
          f"oracleLow={r['oracle_low']:.2f} oracleHigh={r['oracle_high']:.2f} AUCspec={r['auc_spectral']:.3f} AUCpair={r['auc_pairwise']:.3f}")
if "bbp_transfer" in R: print(f"[B5] BBP transfer: {R['bbp_transfer']}")
print("=" * 80)
with open(f"{OUTDIR}/step3_results.json", "w") as fh: json.dump(R, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step3_results.json and step3_figure.png")
