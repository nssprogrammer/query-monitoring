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
# | Prediction log (Step 4), row "One curve, x50 ∈ [1.0, 1.35]" (P4.1) | logistic-fit x50 of clean-λ_min power vs x; configurations classified by x > x50 | Pass (1.052; 47/48) | Part C cell, `x50`, prints "logistic fit: x50" and "configs classified correctly"; VERDICT REPORT "[C]" |
# | Prediction log (Step 4), row "Herding S̄ falls with N" (P4.2) | whitened union S̄ of `herding` as N grows | Fail (rises) | Part B cell, column S for design `herding` across N; VERDICT REPORT "[B]" lines |
# | Prediction log (Step 4), row "Aligned herding detected almost everywhere" (P4.3) | clean-λ_min power of `herding_aligned` vs x | Partial (exactly where x > 1) | Part B cell, columns x and clean for `herding_aligned` |
# | Prediction log (Step 4), row "Spectral linkage AUC ≥ 0.9" (P4.4) | colluder-identification AUC, spectral vs pairwise, where clean λ_min fires | Pass (0.986 vs 0.481) | Part C cell, print "P4.4 linkage where clean_min fires" |
# | Prediction log (Step 4), row "Oracle-only band exists" (P4.5) | power in the band 2.3 < z_oracle < √A, oracle vs spectral | Pass (0.75 vs 0.03) | Part C cell, print "P4.5 oracle-only band" |
#
# Prediction log consequences: "Lower-edge law" (P4.1), "Pool floor" (P4.2), "Appendix A.1" (P4.5). The law tested here
# is the one stated without a number in App. E "Spectral detection condition" (a spectral test needs
# |S̄ − 1|√m ≳ √A, a test told which accounts collude only |S̄ − 1|√m ≳ 1).
#
# Not reported in the paper: per-configuration null thresholds and TW-centering z (Part A), logistic width,
# per-configuration S̄ / z_oracle / x / power values, the herding-backend self-check, step4_figure.png and
# step4_results.json.

# %% [markdown]
# # Step 4 — The linkage price: a unified detectability law in high dimension
# **Kaggle:** GPU → **Run All** (≈30–60 min). `QUICK=True` for a smoke test.
#
# **The law under test.** Rewriting the BBP condition $|\ell|>\sqrt{\gamma}$ with $\ell=\bar S_{\rm union}-1$, $\gamma=A/m_{\rm eff}$ gives
# $$\underbrace{(1-\bar S_{\rm union})\sqrt{m_{\rm eff}}}_{z_{\rm oracle}:\ \text{SNR if colluders were known}}\ \gtrsim\ \underbrace{\sqrt{A}}_{\text{price of unknown linkage}}\quad\Longleftrightarrow\quad x\equiv z_{\rm oracle}/\sqrt{A}\gtrsim 1 .$$
# Steps 1–3 already fall on it (Step 2's 50% point sits at $x=1.13$; Step 3 herding at $x=0.16$–$0.38$, correctly undetected). Step 4 tests it **prospectively** in $d$ = 8 / 32 by moving $x$ across 1 with three independent knobs: campaign size $N$, number of monitored accounts $A$, and attacker–defender alignment.
#
# **Designs.** `herding` = MMD-optimal coreset in the attacker's own Gaussian kernel (realistic). `herding_aligned` = herding directly on the defender's whitened modes (worst case for the attacker, i.e. upper bound on visibility). Colluders are 20% of accounts ($K=A/5$). Thresholds are calibrated per configuration from simulated null worlds ($\alpha$ = 1% per side).
#
# | ID | Locked prediction |
# |---|---|
# | **P4.1** | Clean-λ_min power vs $x$ collapses onto **one curve** across $d$, $h$, $N$, $A$, design, rising through 0.5 at $x_{50}\in[1.0,1.35]$ |
# | **P4.2** | `herding`: $\bar S$ falls with $N$, but stays below the line ($x<1$) at $A=100$; may cross only at $A=25$ with the largest $N$ |
# | **P4.3** | `herding_aligned`: $\bar S\ll1$, $x\gg1$, detected in (almost) all configurations |
# | **P4.4** | Wherever clean-λ_min fires: spectral linkage AUC ≥ 0.9, pairwise-similarity AUC ≈ 0.5 |
# | **P4.5** | Oracle (known-linkage) test detects whenever $z_{\rm oracle}\gtrsim 2.3$, i.e. a band $2.3\lesssim z_{\rm oracle}\lesssim\sqrt{A}$ exists where colluders are detectable **only** if linkage is known |

# %%
# ===== Cell 1: configuration & backend =====
import math, json, time, os
import numpy as np
from scipy.special import airy
from scipy.optimize import brentq, minimize
from scipy.stats import rankdata, gamma as gamma_dist

QUICK = False
SEED = 4041
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
CHUNK = 65536 if USE_GPU else 8192
M_MODES, NCOMP, POOL_MULT, ALPHA = 400, 10, 3, 0.01
ALPHA_SIDE = ALPHA
if QUICK:
    DIMS, HS, AS, NS, TR, N_NULL = [8], {8: [2.0, 4.0]}, [25], [1500], 2, 40
elif USE_GPU:
    DIMS, HS, AS, NS, TR, N_NULL = [8, 32], {8: [2.0, 4.0], 32: [4.0, 8.0]}, [25, 100], [1500, 4000, 12000], 20, 500
else:
    DIMS, HS, AS, NS, TR, N_NULL = [8, 32], {8: [2.0, 4.0], 32: [4.0, 8.0]}, [25, 100], [1500, 4000], 6, 150
A = max(AS)   # placeholder used by legacy helpers
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU/numpy'} | QUICK={QUICK}")
print(f"DIMS={DIMS} HS={HS} AS={AS} NS={NS} TR={TR} N_NULL={N_NULL} (per-config null worlds), alpha/side={ALPHA}")


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
# ===== Cell 3: core (benign model, exact whitening, Fourier, evaluation) =====
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
    me = int(keep.sum()); ok = True
    return dict(h=h, modes=Om / (2 * np.pi), mu=mu, Wk=Wk, meff=me, ok=ok, cond=float(ew.max() / max(ew[keep].min(), 1e-300)),
                tmin=np.nan, tmax=np.nan)

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

def evaluate(pts, lab, cnt, banks, Kc, thr=None):
    # thr: list (per bank) of dict(tmin, tmax, olow) or None -> analytic
    A_ = len(cnt); iu = np.triu_indices(A_, 1); pc = (iu[0] < Kc) & (iu[1] < Kc)
    allmodes = np.vstack([b["modes"] for b in banks]); Fall = fourier_accounts(pts, lab, A_, allmodes)
    sq = np.sqrt(cnt); Ncol = cnt[:Kc].sum(); out = []
    for j, b in enumerate(banks):
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]
        Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / Ncol)
        Fw = Fw / np.sqrt((np.abs(Fw) ** 2).mean(1))[:, None]
        ev, V = np.linalg.eigh(Fw @ Fw.conj().T / b["meff"])
        r = dict(lam_min=ev[0], lam_max=ev[-1], S_union=S_un, z_oracle=(1 - S_un) * math.sqrt(b["meff"]))
        if thr is not None:
            t = thr[j]; nb_, na_ = int((ev < t["tmin"]).sum()), int((ev > t["tmax"]).sum())
            r.update(clean_min=float(nb_ == 1), multi_min=float(nb_ >= 2), max=float(na_ >= 1), oracle_low=float(S_un < t["olow"]))
            v = V[:, 0] if (nb_ >= 1 or na_ == 0) else V[:, -1]          # eigenvector of the outlier that fired
            r["auc_spectral"] = _auc(np.abs(v[:Kc]) ** 2, np.abs(v[Kc:]) ** 2)
            mu_a = Fr / cnt[:, None]; G = mu_a @ mu_a.conj().T; nrm = np.real(np.diag(G))
            D2 = (nrm[:, None] + nrm[None, :] - 2 * np.real(G))[iu]
            r["auc_pairwise"] = _auc(-D2[pc], -D2[~pc])
        out.append(r)
    return out



# %%
# ===== Cell 4: attacker campaigns (herding in attacker's kernel, or aligned with the defender's modes) =====
def _herd_numpy(Phi, N):
    target = Phi.mean(0); base = Phi @ target; Ps = np.zeros(len(Phi)); taken = np.zeros(len(Phi), bool); sel = np.empty(N, int)
    for t in range(N):
        sc = base - Ps / (t + 1); sc[taken] = -np.inf; i = int(np.argmax(sc)); sel[t] = i; taken[i] = True; Ps += Phi @ Phi[i]
    return sel
def _herd_torch(Phi, N):
    P = torch.as_tensor(Phi, dtype=torch.float32, device=DEV); base = P @ P.mean(0)
    Ps = torch.zeros(P.shape[0], dtype=torch.float32, device=DEV); taken = torch.zeros(P.shape[0], dtype=torch.bool, device=DEV)
    sel = torch.empty(N, dtype=torch.long, device=DEV)
    for t in range(N):
        sc = (base - Ps / (t + 1)).masked_fill(taken, float("-inf")); i = torch.argmax(sc)
        sel[t] = i; taken[i] = True; Ps += P @ P[i]
    return sel.cpu().numpy()
def herd(Phi, N): return _herd_torch(Phi, N) if USE_GPU else _herd_numpy(Phi, N)

def feats_attacker_kernel(pool, rng, h_att=1.0, D=256):
    W = rng.standard_normal((pool.shape[1], D)) / h_att; b = rng.uniform(0, 2 * np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(pool @ W + b)
def feats_aligned(pool, bank):
    # single-query whitened mode vectors z(x) = (exp(-i w.x) - mu) Wk  ->  real features [Re z, Im z]
    Z = np.empty((len(pool), bank["Wk"].shape[1]), dtype=np.complex128)
    for s0 in range(0, len(pool), 8192):
        E = np.exp(-2j * np.pi * (pool[s0:s0 + 8192] @ bank["modes"].T)); Z[s0:s0 + 8192] = (E - bank["mu"]) @ bank["Wk"]
    return np.hstack([Z.real, Z.imag])

def make_campaign(gmm, design, N, rng, bank):
    if design == "iid": return gmm.sample(N, rng)
    pool = gmm.sample(POOL_MULT * N, rng)
    Phi = feats_attacker_kernel(pool, rng) if design == "herding" else feats_aligned(pool, bank)
    return pool[herd(Phi, N)]

def world_from_campaign(gmm, camp, A_, rng):
    Kc = A_ // 5; camp = camp[rng.permutation(len(camp))]
    lab_c = rng.integers(0, Kc, len(camp)); n_per = len(camp) // Kc
    ben = gmm.sample((A_ - Kc) * n_per, rng); lab_b = Kc + np.repeat(np.arange(A_ - Kc), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b])
    return pts, lab, np.bincount(lab, minlength=A_), Kc

GMMS = {d: GMM(d, 100 + d) for d in DIMS}
BANKS = {d: [make_bank(GMMS[d], h, 5000 + 10 * d + int(10 * h)) for h in HS[d]] for d in DIMS}
for d in DIMS: print(f"d={d} eff_dim={GMMS[d].eff_dim:.2f} banks: " + ", ".join(f"h={b['h']}: m_eff={b['meff']}" for b in BANKS[d]))

# self-checks: herding backends agree (objective), aligned herding drives the union statistic down
rng = np.random.default_rng(0); pool = GMMS[DIMS[0]].sample(3000, rng); Phi = feats_aligned(pool, BANKS[DIMS[0]][1])
s_np = _herd_numpy(Phi, 300); obj = lambda s: float(np.sum((Phi[s].sum(0)) ** 2))
msg = f"herding objective numpy={obj(s_np):.3f}"
if USE_GPU: msg += f" torch={obj(_herd_torch(Phi, 300)):.3f} (should be ~equal)"
rand = rng.choice(len(pool), 300, replace=False)
print(msg + f" | random subset objective={obj(rand):.1f}  (herding must be far smaller)")
RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, DIMS=DIMS, HS={str(k): v for k, v in HS.items()}, AS=AS, NS=NS, TR=TR,
                          N_NULL=N_NULL, ALPHA=ALPHA, backend="gpu" if USE_GPU else "cpu"),
           "banks": {str(d): [dict(h=b["h"], meff=b["meff"]) for b in BANKS[d]] for d in DIMS}}

# %% [markdown]
# ## Part A — per-configuration null calibration

# %%
# ===== Part A: per-configuration null calibration =====
t0 = time.time(); THR = {}; rowsA = []
for d in DIMS:
    for A_ in AS:
        for N in NS:
            lmn, lmx, Sn = [[] for _ in HS[d]], [[] for _ in HS[d]], [[] for _ in HS[d]]
            for t in range(N_NULL):
                rng = np.random.default_rng(SEED + hash((d, A_, N, t)) % 2**31)
                pts, lab, cnt, Kc = world_from_campaign(GMMS[d], GMMS[d].sample(N, rng), A_, rng)
                for j, r in enumerate(evaluate(pts, lab, cnt, BANKS[d], Kc)):
                    lmn[j].append(r["lam_min"]); lmx[j].append(r["lam_max"]); Sn[j].append(r["S_union"])
            THR[(d, A_, N)] = [dict(tmin=float(np.quantile(lmn[j], ALPHA)), tmax=float(np.quantile(lmx[j], 1 - ALPHA)),
                                    olow=float(np.quantile(Sn[j], ALPHA))) for j in range(len(HS[d]))]
            for j, b in enumerate(BANKS[d]):
                mu_m, sg_m, mu_p, sg_p = edges(A_, b["meff"])
                rowsA.append(dict(d=d, A=A_, N=N, h=b["h"], meff=b["meff"], tmin=THR[(d, A_, N)][j]["tmin"], tmin_TW=mu_m - sg_m * tw_q(1 - ALPHA),
                                  z_min=float((np.mean(lmn[j]) - (mu_m - sg_m * TW_MEAN)) / sg_m)))
            print(f"d={d:2d} A={A_:3d} N={N:5d}: " + " | ".join(f"h={r['h']}: thr_min emp/TW {r['tmin']:.4f}/{r['tmin_TW']:.4f} z={r['z_min']:+.2f}"
                  for r in rowsA[-len(HS[d]):]) + f"  ({time.time()-t0:.0f}s)")
RESULTS["A"] = rowsA

# %% [markdown]
# ## Part B — attacks

# %%
# ===== Part B: attacks =====
t0 = time.time(); rowsB = []
for d in DIMS:
    for N in NS:
        for design in ["herding", "herding_aligned"]:
            acc = {(A_, j): [] for A_ in AS for j in range(len(HS[d]))}
            for t in range(TR):
                rng = np.random.default_rng(SEED + 999 + hash((d, N, design, t)) % 2**31)
                camp = make_campaign(GMMS[d], design, N, rng, BANKS[d][-1])     # aligned attacker targets the lowest-frequency bank
                for A_ in AS:                                                   # same campaign, different monitored populations
                    pts, lab, cnt, Kc = world_from_campaign(GMMS[d], camp, A_, rng)
                    for j, r in enumerate(evaluate(pts, lab, cnt, BANKS[d], Kc, THR[(d, A_, N)])):
                        acc[(A_, j)].append(r)
            for (A_, j), rs in acc.items():
                m_ = lambda k: float(np.mean([r[k] for r in rs])); b = BANKS[d][j]
                row = dict(d=d, N=N, design=design, A=A_, h=b["h"], meff=b["meff"], S_union=m_("S_union"), z_oracle=m_("z_oracle"),
                           x=m_("z_oracle") / math.sqrt(A_), clean_min=m_("clean_min"), multi_min=m_("multi_min"), max=m_("max"),
                           oracle_low=m_("oracle_low"), auc_spectral=m_("auc_spectral"), auc_pairwise=m_("auc_pairwise"), n=len(rs))
                rowsB.append(row)
                # [PAPER] Prediction log (Step 4): herding S̄ vs N (rises: Fail, P4.2); herding_aligned detected exactly where x > 1 (Partial, P4.3)
                print(f"d={d:2d} N={N:5d} {design:>15} A={A_:3d} h={b['h']}: S={row['S_union']:.3f} z_or={row['z_oracle']:6.2f} x={row['x']:5.2f} | "
                      f"clean={row['clean_min']:.2f} multi={row['multi_min']:.2f} max={row['max']:.2f} oracle={row['oracle_low']:.2f} | "
                      f"AUC spec/pair={row['auc_spectral']:.2f}/{row['auc_pairwise']:.2f}  ({time.time()-t0:.0f}s)")
RESULTS["B"] = rowsB

# %% [markdown]
# ## Part C — single-law test, linkage, oracle gap

# %%
# ===== Part C: does everything collapse onto one curve? =====
xs = np.array([r["x"] for r in rowsB]); ys = np.array([r["clean_min"] for r in rowsB]); ns = np.array([r["n"] for r in rowsB])
def nll(p):
    a, s = p; q = 1 / (1 + np.exp(-(xs - a) / max(abs(s), 1e-3))); q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.sum(ns * (ys * np.log(q) + (1 - ys) * np.log(1 - q)))
fitp = minimize(nll, [1.1, 0.1], method="Nelder-Mead").x
x50, width = float(fitp[0]), float(abs(fitp[1]))
pred_ok = [(r["x"] > x50) == (r["clean_min"] > 0.5) for r in rowsB]
# [PAPER] Prediction log (Step 4): one curve, x50 = 1.052, 47/48 configurations (locked range [1.0, 1.35])
print(f"logistic fit: x50 = {x50:.3f}, width = {width:.3f}  (locked prediction: x50 in [1.0, 1.35])")
print(f"configs classified correctly by the single rule x > x50: {np.mean(pred_ok):.3f} ({sum(pred_ok)}/{len(pred_ok)})")
fired = [r for r in rowsB if r["clean_min"] > 0.5]
# [PAPER] Prediction log (Step 4): spectral vs pairwise linkage AUC, 0.986 vs 0.481
if fired: print(f"P4.4 linkage where clean_min fires (n={len(fired)}): spectral AUC {np.mean([r['auc_spectral'] for r in fired]):.3f}, pairwise {np.mean([r['auc_pairwise'] for r in fired]):.3f}")
band = [r for r in rowsB if 2.3 < r["z_oracle"] < math.sqrt(r["A"])]
# [PAPER] Prediction log (Step 4): oracle-only band, oracle 0.75 vs spectral 0.03
if band: print(f"P4.5 oracle-only band (2.3 < z_or < sqrt(A), n={len(band)}): oracle power {np.mean([r['oracle_low'] for r in band]):.2f}, spectral {np.mean([r['clean_min'] for r in band]):.2f}")
RESULTS["C"] = dict(x50=x50, width=width, frac_correct=float(np.mean(pred_ok)))

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
mk = {"herding": "o", "herding_aligned": "s"}; cl = {8: "tab:blue", 32: "tab:red"}
for r in rowsB:
    ax[0].scatter(r["x"], r["clean_min"], marker=mk[r["design"]], color=cl[r["d"]], s=25 + 25 * (r["A"] == 25), alpha=.7)
xx = np.linspace(0, max(xs.max(), 2) * 1.05, 300); ax[0].plot(xx, 1 / (1 + np.exp(-(xx - x50) / max(width, 1e-3))), "k-", lw=1, label=f"logistic fit x50={x50:.2f}")
ax[0].axvline(1, color="gray", ls=":", label="asymptotic BBP (x=1)"); ax[0].set_xscale("symlog", linthresh=0.5)
ax[0].set_xlabel("x = z_oracle / sqrt(A)"); ax[0].set_ylabel("clean λ_min power"); ax[0].set_title("P4.1: one curve? (o herding, □ aligned; blue d=8, red d=32; big = A=25)"); ax[0].legend(fontsize=8)
for d in DIMS:
    for des in ["herding", "herding_aligned"]:
        for A_ in AS:
            rs = sorted([r for r in rowsB if r["d"] == d and r["design"] == des and r["A"] == A_ and r["h"] == HS[d][-1]], key=lambda r: r["N"])
            ax[1].plot([r["N"] for r in rs], [r["x"] for r in rs], marker=mk[des], color=cl[d], ls="-" if A_ == 25 else "--", label=f"d={d} {des} A={A_}")
ax[1].axhline(1, color="gray", ls=":"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("campaign size N")
ax[1].set_ylabel("x"); ax[1].set_title("P4.2/4.3: crossing the line with campaign size"); ax[1].legend(fontsize=6)
ax[2].scatter([r["z_oracle"] for r in rowsB], [r["oracle_low"] for r in rowsB], label="oracle (known colluders)", s=18)
ax[2].scatter([r["z_oracle"] for r in rowsB], [r["clean_min"] for r in rowsB], label="spectral (linkage-free)", s=18, marker="x")
ax[2].axvline(2.33, color="gray", ls=":"); ax[2].set_xscale("symlog", linthresh=1); ax[2].set_xlabel("z_oracle"); ax[2].set_ylabel("power")
ax[2].set_title("P4.5: the linkage gap"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step4_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy this cell's output into the chat

# %%
R = RESULTS
print("=" * 80); print("STEP 4 VERDICT REPORT"); print("=" * 80)
print("config:", json.dumps(R["config"])); print("banks:", R["banks"])
zs = [r["z_min"] for r in R["A"]]; print(f"[A] null lam_min TW-centering z over {len(zs)} configs: mean {np.mean(zs):+.2f}, range [{min(zs):+.2f}, {max(zs):+.2f}]")
for r in R["B"]:
    print(f"[B] d={r['d']:2d} N={r['N']:5d} {r['design']:>15} A={r['A']:3d} h={r['h']}: S={r['S_union']:.3f} z_or={r['z_oracle']:.2f} x={r['x']:.2f} "
          f"clean={r['clean_min']:.2f} multi={r['multi_min']:.2f} max={r['max']:.2f} oracle={r['oracle_low']:.2f} AUC={r['auc_spectral']:.2f}/{r['auc_pairwise']:.2f}")
print(f"[C] x50={R['C']['x50']:.3f} width={R['C']['width']:.3f} single-rule accuracy={R['C']['frac_correct']:.3f}")
print("=" * 80)
with open(f"{OUTDIR}/step4_results.json", "w") as fh: json.dump(R, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step4_results.json and step4_figure.png")
