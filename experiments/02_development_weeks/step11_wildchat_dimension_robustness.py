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
# | App. F "Small monitored populations" | broad capability, 5,000 prompts: paired (attributable) detection from d = 32 to d = 384 | 0.18 to 0.35 | Cell 6 print "P11.3" (`capability`, N = 5000) and verdict "[B]" lines; plausibly A = 100, `rolling_lag` (paper does not name the cell) |
# | App. F "Small monitored populations" | broad, 50% natural padding, 20,000 prompts: paired detection from d = 32 to d = 384 | 0.36 to 0.60 | verdict "[B]" lines, `capability_pad50` N = 20000 (`att`); plausibly A = 100, `rolling_lag` |
# | Prediction log (Step 11, "Lagged calibration nominal at 100, not 500, every d") | `rolling_lag` held-out FPR at A = 100 / A = 500 over d | Pass (3.4–4.4%; 22–39%) | Cell 5 print "held-out FPR"; Cell 6 print "P11.1"; verdict "[A]" lines |
# | Prediction log (Step 11, "Narrow at 5,000 ≥ 0.8 at every d") | narrow capability, N = 5,000, A = 100, paired detection | Pass (0.94–0.96) | Cell 6 print "P11.2" |
# | Prediction log (Step 11, "Broad at 5,000 ≤ 0.35 at every d and A") | broad capability, N = 5,000 | Partial (up to 0.44 at A = 500) | Cell 6 print "P11.3" |
# | Prediction log (Step 11, "Random ≤ 0.05") | max paired detection of random campaigns per d | Pass (≤ 0.01) | Cell 6 print "P11.4" |
# | Prediction log (Step 11, "Padding drops ≥ 0.4 at every d") | drop `capability` → `capability_pad50`, N = 20,000, A = 100, `rolling_lag` | Partial (0.32 at d = 384) | Cell 6 print "P11.5" |
# | App. Table 16 (Embedding) | MiniLM, PCA to d ∈ {32, 64, 128, 384}, rescaled to total variance 4 | as quoted | Cell 1 `DIMS`; Cell 2 `make_Z` |
#
# Not reported in the paper: `rolling` (non-lagged) cells and FPRs except as covered by the ranges above,
# the `excess` columns, the explained-variance printout (the 38% quoted below is not in the paper), d = 64/128
# values individually, and the figure. The d = 32 cells re-run Step 10's design but need not match App. Table 5
# exactly (here the attacker's clusters are fitted in 384-d); Table 5 comes from Step 10.

# %% [markdown]
# # Step 11 — Robustness to the defender's embedding dimension (WildChat-1M)
# **Kaggle:** GPU + Internet + `HF_TOKEN` secret → **Run All** (≈60–80 min: ≈30 min to re-download and embed once, cached as float16; then ≈7–10 min per dimension). `QUICK=True` first.
#
# Step 10 used a 32-dimensional PCA of MiniLM that keeps only **38%** of the embedding variance. Step 11 repeats Step 10's key cells with the defender's projection at **d = 32, 64, 128 and 384 (no reduction)**. Everything else is held fixed and paired across d: the same accounts, weeks, test windows (same seeds), and the same campaigns. The attacker's capability clusters are now defined in the **full 384-d space** (k-means with 10 and 30 clusters on the attacker's pool), so they do not depend on the defender's view. Each projection is rescaled to total variance 4, so the bandwidth h = 4 keeps the same meaning.
#
# Schemes: `rolling` and `rolling_lag` (Step 10's two drift-aware schemes; overall α = 5%). Metric: paired **attributable power**.
#
# | ID | Locked prediction |
# |---|---|
# | **P11.1** | `rolling_lag` held-out FPR ≤ 7.5% at A = 100 for every d; > 10% at A = 500 for every d (the A = 500 failure is not a projection artefact) |
# | **P11.2** | Narrow capability, N = 5,000, A = 100: attributable power ≥ 0.8 for every d (both schemes) |
# | **P11.3** | Broad capability, N = 5,000: attributable power ≤ 0.35 for every d and both A (the broad/narrow contrast is not an artefact of discarding 62% of the variance) |
# | **P11.4** | Random campaigns: attributable power ≤ 0.05 for every d |
# | **P11.5** | 50% padding at N = 20,000, A = 100: attributable power drops by ≥ 0.4 for every d under `rolling_lag` |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc
import numpy as np
from scipy.stats import rankdata
QUICK = False
SYNTHETIC_TEST = False          # True only for offline plumbing tests
SEED = 9091                     # same seed as Steps 9-10: same IP split and windows
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
if not USE_GPU and not SYNTHETIC_TEST:
    raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
CHUNK = 32768 if USE_GPU else 8192
H_BW, M_MODES, ALPHA, FLOOR = 4.0, 400, 0.05, 1e-4
ALPHA_D = ALPHA / 3
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, CAL_FRAC = 500, 3, 0.2, 0.6
STRATS = ["random", "capability", "capability_narrow", "capability_pad50"]; SCHEMES = ["rolling", "rolling_lag"]
if QUICK:
    N_CONV, DIMS, AK, NS, TRIALS, N_TW, N_CAL_ROLL, N_NULL_TW, W_TW, LAG = 30000, [32, 128], [(50, 10)], [300, 1000], 1, 2, 40, 10, 2, 2
else:
    # [PAPER] App. Table 16: PCA to d ∈ {32, 64, 128, 384}; App. F: "from 32 to 384 dimensions"
    N_CONV, DIMS, AK, NS, TRIALS, N_TW, N_CAL_ROLL, N_NULL_TW, W_TW, LAG = None, [32, 64, 128, 384], [(100, 20), (500, 100)], [1000, 5000, 20000], 3, 8, 250, 40, 3, 4
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} | DIMS={DIMS} | cache {CACHE11}")

# %% [markdown]
# ## Part 0 — 384-d embeddings (cached)

# %%
# ===== Cell 2: full 384-d MiniLM embeddings of English user turns (cached as float16) =====
if os.path.exists(CACHE11):
    _c = np.load(CACHE11); E16, ACCT, WEEK = _c["E16"], _c["ACCT"], _c["WEEK"]; print(f"loaded cache: {E16.shape}")
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        rng = np.random.default_rng(0); n_ip, n_rows = 3000, 60000
        ACCT = rng.integers(0, n_ip, n_rows).astype(np.int32); WEEK = rng.integers(0, 30, n_rows).astype(np.int16)
        topic = rng.integers(0, 40, n_ip)[ACCT]; base = rng.standard_normal((40, 384))
        E = (base[topic] * 0.8 + rng.standard_normal((n_rows, 384)) * 0.6).astype(np.float32)
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        split = "train" if N_CONV is None else f"train[:{N_CONV}]"
        ds = load_dataset("allenai/WildChat-1M", split=split, token=tok).select_columns(["conversation", "hashed_ip", "timestamp", "language"])
        texts, ips, tss = [], [], []
        for batch in ds.iter(batch_size=2000):
            for conv, ip, ts, lang in zip(batch["conversation"], batch["hashed_ip"], batch["timestamp"], batch["language"]):
                if lang != "English" or ip is None or ts is None: continue
                for turn in conv:
                    if turn.get("role") == "user" and turn.get("content"):
                        texts.append(turn["content"][:1000]); ips.append(ip); tss.append(ts)
        print(f"{len(texts)} English user turns from {len(set(ips))} IPs ({time.time()-t0:.0f}s)")
        _, ACCT = np.unique(np.array(ips), return_inverse=True); ACCT = ACCT.astype(np.int32)
        days = np.array([np.datetime64(t.replace(tzinfo=None) if hasattr(t, "replace") else t, "D") for t in tss])
        WEEK = ((days - days.min()).astype(np.int64) // 7).astype(np.int16); del ds; gc.collect()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        E = enc.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        del texts; gc.collect()
    E16 = E.astype(np.float16); del E; gc.collect()
    np.savez(CACHE11, E16=E16, ACCT=ACCT, WEEK=WEEK); print(f"cached ({time.time()-t0:.0f}s)")
n_w = int(WEEK.max()) + 1
# PCA basis on a 100k fit sample (nested: the first d components of the same basis for every d)
rng = np.random.default_rng(SEED); fit = E16[rng.choice(len(E16), min(100000, len(E16)), replace=False)].astype(np.float32)
MU_E = fit.mean(0); ew, U = np.linalg.eigh(np.cov(fit - MU_E, rowvar=False)); EW, UB = ew[::-1], U[:, ::-1].copy()
for d in DIMS: print(f"d={d:3d}: explains {EW[:d].sum()/EW.sum():.3f} of embedding variance")
def make_Z(d):
    W_ = UB[:, :d] / math.sqrt(EW[:d].sum() / 4.0); Z = np.empty((len(E16), d), dtype=np.float32)
    for s0 in range(0, len(E16), 200000): Z[s0:s0 + 200000] = (E16[s0:s0 + 200000].astype(np.float32) - MU_E) @ W_
    return Z


# %% [markdown]
# ## Part 1 — accounts, weeks, campaigns

# %%
# ===== Cell 3: accounts, weeks, attacker pool, campaigns (independent of the defender's d) =====
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC
cal_weeks = np.arange(int(round(CAL_FRAC * n_w))); test_weeks = np.arange(len(cal_weeks), n_w)
order = np.lexsort((ACCT, WEEK)); keys = WEEK[order].astype(np.int64) * n_acct + ACCT[order]
groups = np.split(order, np.flatnonzero(np.diff(keys)) + 1); WEEK_ACCTS = {w: [] for w in range(n_w)}
for g in groups:
    a = int(ACCT[g[0]]); w = int(WEEK[g[0]])
    if is_pool[a] or len(g) < MIN_PER_ACCT: continue
    if len(g) > VOL_CAP: g = rng.choice(g, VOL_CAP, replace=False)
    WEEK_ACCTS[w].append(g)
pool_idx = np.flatnonzero(is_pool[ACCT])
ref_idx = np.concatenate([np.concatenate(WEEK_ACCTS[w]) for w in cal_weeks if WEEK_ACCTS[w]])
okw = [w for w in test_weeks if len(WEEK_ACCTS[w]) >= max(a for a, _ in AK) and w - LAG >= 0]
TW = [int(okw[i]) for i in np.linspace(0, len(okw) - 1, min(N_TW, len(okw))).astype(int)] if okw else []
print(f"pool {len(pool_idx)} queries | reference {len(ref_idx)} | test weeks {TW} (eligible: {[len(WEEK_ACCTS[w]) for w in TW]})")

from sklearn.cluster import MiniBatchKMeans
t0 = time.time(); Ep = E16[pool_idx].astype(np.float32)
CL10 = MiniBatchKMeans(n_clusters=10, random_state=SEED, batch_size=4096, n_init=3).fit(Ep).labels_
CL30 = MiniBatchKMeans(n_clusters=30, random_state=SEED + 1, batch_size=4096, n_init=3).fit(Ep).labels_; del Ep
s10, s30 = np.bincount(CL10, minlength=10), np.bincount(CL30, minlength=30)
print(f"384-d clusters: broad sizes {sorted(s10.tolist(), reverse=True)}; narrow sizes min/median/max {s30.min()}/{int(np.median(s30))}/{s30.max()} ({time.time()-t0:.0f}s)")
Nmax = max(NS); CAMP = {}                       # indices into pool_idx
for tr_ in range(TRIALS):
    r = np.random.default_rng(SEED + 100 + tr_); b = np.flatnonzero(CL10 == int(np.argsort(-s10)[tr_ % 5]))
    ok30 = np.flatnonzero(s30 >= sorted(NS)[len(NS) // 2]); c30 = int(ok30[tr_ % len(ok30)]) if len(ok30) else int(np.argmax(s30)); n_ = np.flatnonzero(CL30 == c30)
    CAMP[("random", tr_)] = r.choice(len(pool_idx), Nmax, replace=False)
    CAMP[("capability", tr_)] = b[r.choice(len(b), Nmax, replace=len(b) < Nmax)]
    CAMP[("capability_narrow", tr_)] = n_[r.permutation(len(n_))]
    CAMP[("capability_pad50", tr_)] = np.where(np.arange(Nmax) % 2 == 0, CAMP[("capability", tr_)], r.choice(len(pool_idx), Nmax, replace=False))
    print(f"trial {tr_}: broad cluster {len(b)} prompts, narrow cluster {len(n_)} prompts")


# %%
# ===== Cell 4: whitening banks, statistics, calibration =====
def make_bank(Zref, seed):
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / H_BW / (2 * np.pi); m = M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
        Gt = torch.zeros((m, m), dtype=torch.complex128, device=DEV); mut = torch.zeros(m, dtype=torch.complex128, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            E_ = torch.complex(torch.cos(th), -torch.sin(th)); Gt += E_.T @ E_.conj(); mut += E_.sum(0)
        G, mu = Gt.cpu().numpy(), mut.cpu().numpy()
    else:
        G = np.zeros((m, m), dtype=np.complex128); mu = np.zeros(m, dtype=np.complex128)
        for s0 in range(0, len(Zref), 4096):
            E_ = np.exp(-2j * np.pi * (Zref[s0:s0 + 4096] @ modes.T)); G += E_.T @ E_.conj(); mu += E_.sum(0)
    G /= len(Zref); mu /= len(Zref); Sig = G - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew_, U_ = np.linalg.eigh(Sig); keep = ew_ > FLOOR * ew_.max()
    B = dict(modes=modes, mu=mu, Wk=U_[:, keep].conj() / np.sqrt(ew_[keep]), meff=int(keep.sum()))
    if USE_GPU:
        B["MT"] = torch.as_tensor(modes, dtype=torch.float64, device=DEV); B["MU_T"] = torch.as_tensor(mu, dtype=torch.complex128, device=DEV)
        B["WK_T"] = torch.as_tensor(B["Wk"], dtype=torch.complex128, device=DEV)
    return B
def stats_b(pts, lab, cnt, B):
    if USE_GPU:
        Fr = torch.zeros((len(cnt), M_MODES), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
        for s0 in range(0, len(pts), CHUNK):
            P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV); ix = torch.as_tensor(lab[s0:s0 + CHUNK], dtype=torch.long, device=DEV)
            th = (2.0 * math.pi) * (P @ B["MT"].T); Fr.index_add_(0, ix, torch.cos(th)); Fi.index_add_(0, ix, -torch.sin(th))
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV); sq = torch.sqrt(c)
        Fw = ((torch.complex(Fr, Fi) - c[:, None] * B["MU_T"][None, :]) / sq[:, None]) @ B["WK_T"]
        return dict(lm=float(torch.linalg.eigvalsh(Fw.conj().T @ Fw / B["meff"])[-1]), r2=(Fw.abs() ** 2).mean(1).cpu().numpy(),
                    ga=float(((Fw * sq[:, None]).sum(0).abs() ** 2).mean() / c.sum()))
    import scipy.sparse as sp
    F_ = np.zeros((len(cnt), M_MODES), dtype=np.complex128)
    for s0 in range(0, len(pts), CHUNK):
        P, l_ = pts[s0:s0 + CHUNK], lab[s0:s0 + CHUNK]; E_ = np.exp(-2j * np.pi * (P @ B["modes"].T))
        F_ += sp.csr_matrix((np.ones(len(P)), (l_, np.arange(len(P)))), shape=(len(cnt), len(P))) @ E_
    sq = np.sqrt(cnt); Fw = ((F_ - cnt[:, None] * B["mu"][None, :]) / sq[:, None]) @ B["Wk"]
    return dict(lm=float(np.linalg.eigvalsh(Fw.conj().T @ Fw / B["meff"])[-1]), r2=(np.abs(Fw) ** 2).mean(1), ga=float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum()))
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))
def calibrate(recs):
    q_ = lambda a: float(np.quantile(a, 1 - ALPHA_D)); ns = np.sort(np.concatenate([r["r2"] for r in recs]))
    return dict(tmax=q_([r["lm"] for r in recs]), tglo=q_([r["ga"] for r in recs]), thc=q_([hc_stat(r["r2"], ns) for r in recs]), ns=ns)
def suite(g, T_): return (g["lm"] > T_["tmax"]) or (hc_stat(g["r2"], T_["ns"]) > T_["thc"]) or (g["ga"] > T_["tglo"])
def window_at(w, A_, r):
    pick = r.choice(len(WEEK_ACCTS[w]), A_, replace=False); parts = [WEEK_ACCTS[w][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(A_), vols)


# %% [markdown]
# ## Part 2 — Step 10 design at d = 32, 64, 128, 384

# %%
# ===== Cell 5: run Step 10's design at every defender dimension d =====
rowsA, rowsB = [], []; T0 = time.time()
for d in DIMS:
    t0 = time.time(); Z = make_Z(d); BANK = make_bank(Z[ref_idx], SEED + 7); LAGB = {}
    def lagb(w):
        if w not in LAGB:
            parts = [np.concatenate(WEEK_ACCTS[v]) for v in range(w - LAG, w) if v >= 0 and WEEK_ACCTS[v]]
            idx = np.concatenate(parts) if parts else None
            if idx is not None and len(idx) > 100000: idx = np.random.default_rng(5).choice(idx, 100000, replace=False)
            LAGB[w] = make_bank(Z[idx], SEED + 7) if idx is not None and len(idx) > 2000 else BANK
        return LAGB[w]
    def ws(idx, lab, A_, w, scheme, extra=None):
        pts = Z[idx] if extra is None else np.vstack([extra[0], Z[idx]]); labs = lab if extra is None else np.concatenate([extra[1], lab])
        return stats_b(pts, labs, np.bincount(labs, minlength=A_), lagb(w) if scheme == "rolling_lag" else BANK)
    print(f"--- d={d}: m_eff={BANK['meff']} ({time.time()-t0:.0f}s)")
    THR = {}
    for A_, K_ in AK:
        for tw in TW:
            prev = [w for w in range(tw - LAG, tw) if len(WEEK_ACCTS[w]) >= A_]
            for s in SCHEMES:
                recs = []
                for t in range(N_CAL_ROLL):
                    r = np.random.default_rng(SEED + 6_000_000 + 1000 * tw + 10 * A_ + t + (7 if s == "rolling_lag" else 0)); w = int(r.choice(prev))
                    idx, lab = window_at(w, A_, r); recs.append(ws(idx, lab, A_, w, s))
                THR[(s, A_, tw)] = calibrate(recs)
        for s in SCHEMES:
            fp = n = 0
            for tw in TW:
                for t in range(N_NULL_TW):
                    r = np.random.default_rng(SEED + 7_000_000 + 1000 * tw + 10 * A_ + t); idx, lab = window_at(tw, A_, r)
                    fp += int(suite(ws(idx, lab, A_, tw, s), THR[(s, A_, tw)])); n += 1
            # [PAPER] Prediction log (Step 11): rolling_lag FPR 3.4–4.4% at A = 100, 22–39% at A = 500 (across d)
            rowsA.append(dict(d=d, A=A_, scheme=s, fpr=fp / n, n=n)); print(f"d={d:3d} A={A_:4d} {s:>11}: held-out FPR {fp/n:.3f}  ({time.time()-t0:.0f}s)")
        for name in STRATS:
            for N in NS:
                if name == "capability_narrow" and N > min(len(CAMP[(name, t)]) for t in range(TRIALS)): continue
                agg = {s: [0, 0] for s in SCHEMES}; n = 0
                for tw in TW:
                    for tr_ in range(TRIALS):
                        campZ = Z[pool_idx[CAMP[(name, tr_)][:N]]]
                        for w in range(W_TW):
                            r = np.random.default_rng(SEED + 8_000_000 + 1000 * tw + 100 * tr_ + w + A_ + N); idx, lab = window_at(tw, A_, r)
                            keep = lab >= K_; n_per = N // K_; extra = (campZ[r.permutation(N)][: K_ * n_per], np.arange(K_ * n_per) % K_); n += 1
                            for s in SCHEMES:
                                T_ = THR[(s, A_, tw)]; base = suite(ws(idx, lab, A_, tw, s), T_); att = suite(ws(idx[keep], lab[keep], A_, tw, s, extra), T_)
                                agg[s][0] += int(att and not base); agg[s][1] += int(att) - int(base)
                row = dict(d=d, A=A_, K=K_, strategy=name, N=N, n=n, **{f"{s}_attrib": agg[s][0] / n for s in SCHEMES}, **{f"{s}_excess": agg[s][1] / n for s in SCHEMES})
                rowsB.append(row)
                # [PAPER] App. F: broad 5,000 from 0.18 to 0.35, padded 20,000 from 0.36 to 0.60 (d = 32 → 384; plausibly A = 100, rolling_lag)
                print(f"d={d:3d} A={A_:4d} {name:>18} N={N:6d} | " + " ".join(f"{s} attrib {row[s+'_attrib']:.2f}" for s in SCHEMES) + f"  ({time.time()-t0:.0f}s)")
    del Z, BANK, LAGB; gc.collect()
    if USE_GPU: torch.cuda.empty_cache()
print(f"all dimensions done in {time.time()-T0:.0f}s")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 6: predictions, figure, verdict =====
g = lambda d, A_, s_, N: next((r for r in rowsB if r["d"] == d and r["A"] == A_ and r["strategy"] == s_ and r["N"] == N), None)
mid, top = sorted(NS)[len(NS) // 2], max(NS); A1 = AK[0][0]
# [PAPER] Prediction log (Step 11): P11.1 Pass (3.4–4.4%; 22–39%)
print("P11.1 rolling_lag FPR: " + "; ".join(f"d={r['d']} A={r['A']}: {r['fpr']:.3f}" for r in rowsA if r["scheme"] == "rolling_lag"))
# [PAPER] Prediction log (Step 11): P11.2 Pass (0.94–0.96)
print("P11.2 narrow N=%d A=%d: " % (mid, A1) + "; ".join(f"d={d}: " + "/".join(f"{g(d, A1, 'capability_narrow', mid)[s+'_attrib']:.2f}" for s in SCHEMES) for d in DIMS if g(d, A1, "capability_narrow", mid)))
# [PAPER] Prediction log (Step 11): P11.3 Partial (up to 0.44 at A = 500); App. F: broad 5,000 from 0.18 to 0.35
print("P11.3 broad N=%d: " % mid + "; ".join(f"d={d} A={A_}: " + "/".join(f"{g(d, A_, 'capability', mid)[s+'_attrib']:.2f}" for s in SCHEMES) for d in DIMS for A_, _ in AK if g(d, A_, "capability", mid)))
# [PAPER] Prediction log (Step 11): P11.4 Pass (≤ 0.01)
print("P11.4 random max attrib per d: " + "; ".join(f"d={d}: {max(max(r[s+'_attrib'] for s in SCHEMES) for r in rowsB if r['d'] == d and r['strategy'] == 'random'):.2f}" for d in DIMS))
# [PAPER] Prediction log (Step 11): P11.5 Partial (0.32 at d = 384)
print("P11.5 pad50 drop at N=%d A=%d (rolling_lag): " % (top, A1) + "; ".join(f"d={d}: {g(d, A1, 'capability', top)['rolling_lag_attrib'] - g(d, A1, 'capability_pad50', top)['rolling_lag_attrib']:+.2f}" for d in DIMS if g(d, A1, "capability", top)))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(AK), figsize=(8 * len(AK), 5), squeeze=False)
for i, (A_, K_) in enumerate(AK):
    for (s_, N), c_ in [((("capability_narrow", mid)), "tab:green"), (("capability", mid), "tab:orange"), (("capability", top), "tab:red"), (("capability_pad50", top), "tab:purple"), (("random", top), "tab:gray")]:
        rr = [g(d, A_, s_, N) for d in DIMS]
        if all(rr): ax[0, i].plot(DIMS, [r["rolling_lag_attrib"] for r in rr], "o-", color=c_, label=f"{s_} N={N}")
    fpr = [next(r["fpr"] for r in rowsA if r["d"] == d and r["A"] == A_ and r["scheme"] == "rolling_lag") for d in DIMS]
    ax[0, i].plot(DIMS, fpr, "k:", marker="x", label="held-out FPR (rolling_lag)")
    ax[0, i].set_xscale("log", base=2); ax[0, i].set_xticks(DIMS); ax[0, i].set_xticklabels(DIMS); ax[0, i].set_ylim(-0.03, 1.03)
    ax[0, i].set_xlabel("defender projection dimension d"); ax[0, i].set_ylabel("attributable power (rolling_lag)"); ax[0, i].set_title(f"A={A_}, K={K_}"); ax[0, i].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step11_figure.png", dpi=130); plt.show()
print("=" * 86); print("STEP 11 VERDICT REPORT"); print("=" * 86)
print("config:", json.dumps(dict(QUICK=QUICK, DIMS=DIMS, AK=AK, NS=NS, TW=TW, ALPHA=ALPHA, LAG=LAG)))
print("explained variance:", {d: round(float(EW[:d].sum() / EW.sum()), 3) for d in DIMS})
for r in rowsA: print(f"[A] d={r['d']:3d} A={r['A']:4d} {r['scheme']:>11} FPR={r['fpr']:.3f}")
# [PAPER] App. F: padded 20,000 (capability_pad50) from 0.36 to 0.60 across d (read from these "[B]" lines)
for r in rowsB: print(f"[B] d={r['d']:3d} A={r['A']:4d} {r['strategy']:>18} N={r['N']:6d} " + " ".join(f"{s}: att {r[s+'_attrib']:.2f} exc {r[s+'_excess']:+.2f}" for s in SCHEMES))
print("=" * 86)
with open(f"{OUTDIR}/step11_results.json", "w") as fh: json.dump(dict(A=rowsA, B=rowsB), fh, indent=1, default=float)
