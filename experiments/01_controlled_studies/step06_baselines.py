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
# | App. P (Step 6), row "Spectral vs global crossover at K ≈ 2.3 √A" (P6.1) | informative cells where the predicted winner (spectral below, global above the crossover) wins | Pass (7 of 8 cells; 2/2) | Part 5 print "P6.1 informative cells" |
# | App. E 'CIFAR-10 regime map' | regime map: spectral dominates when many colluders send few queries each and K ≲ 2.3 √A; per-account tests and Higher Criticism suffice with few heavy colluders; global stream wins with many colluders | 2.3 | Part 5 head-to-head table (`winner` column) / VERDICT `[d]` lines |
# | App. P (Step 6), row "HC ≥ per-account; spectral ≥ HC at N = 2,000" (P6.2) | cell counts | Pass (29/30; 8/10) | Part 5 print "P6.2 HC≥acct in ... cells; spectral≥HC ..." |
# | App. P (Step 6), row "Random campaigns ≤ 2%; linkage AUC ≈ 0.5" (P6.3) | random-campaign false alarms; spectral linkage AUC | Pass (1.9%; 0.49–0.51) | Part 5 prints "P6.3 random-campaign false alarms" and "spectral linkage AUC on random campaigns by A" |
# | App. P (Step 6), row "No real strategy triggers the lower edge" (P6.4); App. E 'Spectral detection condition' ("No real extraction strategy we tested triggered the lower edge") | missing-mode rate on real strategies | Pass (0.002) | Part 5 print "P6.4 missing-mode rate" |
# | Table 16 (App. P, "Main settings"), CIFAR-10 rows | bandwidths with Bonferroni correction; calibration | h ∈ {2, 4, 8} with Bonferroni correction; 1,000 null worlds per configuration, Bonferroni over bandwidths | Cell 1 configuration (`HS`, `N_NULL`); Cell 5 `ALPHA_B` |
#
# Not reported in the paper: fidelity values (re-computed from Step 5), per-cell powers with Wilson intervals, per-account
# linkage AUC, the low-side global test (`glob_lo`), herding cells individually and the figure `step6_figure.png`.

# %% [markdown]
# # Step 6 — Baselines that decide the practical claim (CIFAR-10, real extraction)
# **Kaggle:** GPU + Internet → **Run All** (≈45–70 min; reuses `step5_cache_full.npz` if present). `QUICK=True` for a smoke test.
#
# Step 5b showed spectral λ_max beats *per-account* monitoring when each colluder sends few queries. Two stronger **linkage-free** baselines remain:
# * **Global stream test** — whitened discrepancy of the *entire* traffic window vs the benign reference (the traffic-window distribution test from the recent literature).
# * **Higher Criticism (HC)** over per-account discrepancy p-values — the classical optimal detector for a sparse set of weakly anomalous accounts.
#
# **Theory.** For colluders sending an equal share of traffic, global detection needs $f(\bar S-1)\sqrt{m}\gtrsim 2.3$ with traffic fraction $f=K/A$, while spectral detection needs $(\bar S-1)\sqrt{m}/\sqrt A\gtrsim1$. Hence **spectral beats global iff $K\lesssim 2.3\sqrt A$** — few colluders among many accounts.
#
# **Fixes from 5b:** colluder accounts now have *exactly* equal sizes (the multinomial split leaked account size into the A=400 linkage AUC); every detector is combined over the 3 bandwidths with Bonferroni thresholds (α/3 per bandwidth, 1% overall), null-calibrated with 1000 worlds; 95% Wilson intervals reported.
#
# | ID | Locked prediction |
# |---|---|
# | **P6.1** | Spectral λ_max ≥ global at (A,K) = (25,5), (100,5), (400,20) [K < 2.3√A]; global ≥ spectral at (400,80) [K > 2.3√A]; (100,20) is near the crossover |
# | **P6.2** | HC ≥ per-account max everywhere; spectral ≥ HC for kcenter/entropy at N=2000 |
# | **P6.3** | Random campaigns: every detector fires ≤ 2%; spectral linkage AUC ≈ 0.5 at every A (artifact removed) |
# | **P6.4** | No real strategy triggers the missing mode (rate ≈ α) |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os
import numpy as np
from scipy.stats import rankdata, spearmanr
QUICK = False
SEED = 5051
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
ALLOW_CPU = False   # set True only for a CPU smoke test
if not USE_GPU and not ALLOW_CPU:
    raise RuntimeError("No GPU visible to PyTorch. Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart the session.")
CHUNK = 65536 if USE_GPU else 8192
# [PAPER] Table 16 (CIFAR-10 rows): h ∈ {2, 4, 8} with Bonferroni correction; 1,000 null worlds per configuration (N_NULL below)
D_DEF, HS, M_MODES, ALPHA, FLOOR = 32, [2.0, 4.0, 8.0], 400, 0.01, 1e-4
STRATS = ["random", "entropy", "kcenter", "herding"]
if QUICK:
    EPOCHS, N_REF, N_NULLB, POOL, N_FIT = 2, 40000, 30000, 12000, 10000
    SEED_N, TARGETS, CHECKS, TRIALS, N_NULL, W = 300, [600, 1000], [600, 1000], 1, 60, 2
    AK = [(25, 5), (100, 20)]
else:
    EPOCHS, N_REF, N_NULLB, POOL, N_FIT = 15, 300000, 200000, 60000, 50000
    SEED_N, TARGETS, CHECKS, TRIALS, N_NULL, W = 1000, [2000, 3500, 5000], [2000, 5000], 3, 1000, 12
    AK = [(25, 5), (100, 5), (100, 20), (400, 20), (400, 80)]
AS = sorted({a for a, _ in AK})
CACHE = f"{OUTDIR}/step5_cache_{'quick' if QUICK else 'full'}.npz"
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} | cache={CACHE}")

# %% [markdown]
# ## Part 0 — victim & embeddings (cached from Step 5)

# %%
# ===== Cell 2: data, victim, attacker encoder, embeddings =====
if os.path.exists(CACHE):
    _c = np.load(CACHE); globals().update({k: _c[k] for k in _c.files}); VICTIM_ACC = float(VICTIM_ACC)
    print(f"loaded cache: victim acc {VICTIM_ACC:.4f}, Z_ref {Z_ref.shape}, Z_pool {Z_pool.shape}")
else:
    import torch.nn as nn, torch.nn.functional as F, torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    torch.backends.cudnn.benchmark = True
    t0 = time.time(); DATA = os.path.join(OUTDIR, "data")
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True); te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    X_tr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous().to(DEV); y_tr = torch.tensor(tr.targets, device=DEV)
    X_te = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous().to(DEV); y_te = np.array(te.targets)
    VICTIM_IDX, BASES = np.arange(25000), np.arange(25000, 50000)
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616], device=DEV).view(1, 3, 1, 1)
    IMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); ISTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    AR = torch.arange(32, device=DEV)

    def make_batch(X, base, ox, oy, flip):     # reflect-pad 4, crop 32x32 at (oy, ox), optional h-flip -> float in [0,1]
        xb = X[torch.as_tensor(base, device=DEV, dtype=torch.long)].float()
        pad = F.pad(xb, (4, 4, 4, 4), mode="reflect"); B = xb.shape[0]
        rows = torch.as_tensor(oy, device=DEV, dtype=torch.long)[:, None] + AR[None, :]
        cols = torch.as_tensor(ox, device=DEV, dtype=torch.long)[:, None] + AR[None, :]
        bi = torch.arange(B, device=DEV)[:, None, None, None]; ci = torch.arange(3, device=DEV)[None, :, None, None]
        out = pad[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]
        f = torch.as_tensor(np.asarray(flip), device=DEV, dtype=torch.bool)
        return torch.where(f[:, None, None, None], out.flip(-1), out) / 255.0
    def draw(n, rng, bases):
        return dict(base=rng.choice(bases, n), ox=rng.integers(0, 9, n), oy=rng.integers(0, 9, n), flip=rng.random(n) < 0.5)
    def victim_net():
        m = resnet18(num_classes=10); m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity(); return m
    def victim_feats(m, x):
        x = m.relu(m.bn1(m.conv1(x))); x = m.maxpool(x)
        x = m.layer4(m.layer3(m.layer2(m.layer1(x)))); return torch.flatten(m.avgpool(x), 1)

    # sanity: un-augmented crop reproduces the original image
    chk = make_batch(X_te, np.arange(4), np.full(4, 4), np.full(4, 4), np.zeros(4, bool))
    assert torch.allclose(chk * 255.0, X_te[:4].float()), "augmentation crop check failed"

    # ---- train victim
    rng = np.random.default_rng(SEED)
    victim = victim_net().to(DEV).to(memory_format=torch.channels_last)
    BS = 256; spe = len(VICTIM_IDX) // BS
    opt = torch.optim.SGD(victim.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.2, total_steps=EPOCHS * spe, pct_start=0.25)
    try: scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)
    except Exception: scaler = torch.cuda.amp.GradScaler(enabled=USE_GPU)
    @torch.no_grad()
    def test_acc():
        victim.eval(); pr = []
        for s in range(0, 10000, 1000):
            x = make_batch(X_te, np.arange(s, s + 1000), np.full(1000, 4), np.full(1000, 4), np.zeros(1000, bool))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                pr.append(victim(((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)).argmax(1).cpu().numpy())
        return float((np.concatenate(pr) == y_te).mean())
    for ep in range(EPOCHS):
        victim.train(); perm = rng.permutation(VICTIM_IDX)
        for s in range(spe):
            idx = perm[s * BS:(s + 1) * BS]
            xb = ((make_batch(X_tr, idx, rng.integers(0, 9, BS), rng.integers(0, 9, BS), rng.random(BS) < 0.5) - MEAN) / STD).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                loss = F.cross_entropy(victim(xb), y_tr[torch.as_tensor(idx, device=DEV)], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if ep == EPOCHS - 1 or ep % 5 == 0: print(f"epoch {ep+1}/{EPOCHS} loss {loss.item():.3f} test acc {test_acc():.4f} ({time.time()-t0:.0f}s)")
    VICTIM_ACC = test_acc(); victim.eval()

    # ---- attacker's own encoder (ImageNet ResNet-18)
    enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); enc.fc = nn.Identity(); enc = enc.to(DEV).eval()
    PCA_MU = PCA_W = None
    @torch.no_grad()
    def extract(X, q, want_def=True, want_att=False, bs=1024):
        n = len(q["base"]); D, Lb, At = [], [], []
        for s in range(0, n, bs):
            sl = slice(s, s + bs); x01 = make_batch(X, q["base"][sl], q["ox"][sl], q["oy"][sl], q["flip"][sl])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                if want_def:
                    f = victim_feats(victim, ((x01 - MEAN) / STD).contiguous(memory_format=torch.channels_last))
                    Lb.append(victim.fc(f).argmax(1).cpu().numpy())
                if want_att:
                    xi = (F.interpolate(x01, size=64, mode="bilinear", align_corners=False) - IMEAN) / ISTD
                    At.append(enc(xi).float().cpu().numpy())
            if want_def:                                   # projection in full fp32, outside autocast
                f = f.float(); D.append(((f - PCA_MU) @ PCA_W).cpu().numpy() if PCA_W is not None else f.cpu().numpy())
        cat = lambda L: np.concatenate(L) if L else None
        return cat(D), cat(Lb), cat(At)

    # ---- defender PCA (victim penultimate -> 32 dims, total variance 4)
    Ffit, _, _ = extract(X_tr, draw(N_FIT, rng, BASES))
    mu = Ffit.mean(0); ew, U = np.linalg.eigh(np.cov(Ffit - mu, rowvar=False)); Wp = U[:, ::-1][:, :D_DEF]
    Zf = (Ffit - mu) @ Wp; Wp = Wp / math.sqrt(Zf.var(0).sum() / 4.0)
    PCA_MU = torch.as_tensor(mu, device=DEV, dtype=torch.float32); PCA_W = torch.as_tensor(Wp.copy(), device=DEV, dtype=torch.float32)
    print(f"PCA-32 explains {ew[::-1][:D_DEF].sum()/ew.sum():.3f} of victim-feature variance")

    Z_ref, _, _ = extract(X_tr, draw(N_REF, rng, BASES))
    Z_null, _, _ = extract(X_tr, draw(N_NULLB, rng, BASES))
    Z_pool, vlab_pool, A_pool_raw = extract(X_tr, draw(POOL, rng, BASES), want_att=True)
    qt = dict(base=np.arange(10000), ox=np.full(10000, 4), oy=np.full(10000, 4), flip=np.zeros(10000, bool))
    _, vlab_test, A_test_raw = extract(X_te, qt, want_def=True, want_att=True)
    ytest_true = y_te
    # attacker-side PCA-128 + standardisation (fit on the attacker's own pool)
    am = A_pool_raw.mean(0); aew, aU = np.linalg.eigh(np.cov(A_pool_raw - am, rowvar=False)); aW = aU[:, ::-1][:, :128]
    A_pool = (A_pool_raw - am) @ aW; asd = A_pool.std(0); A_pool = A_pool / asd; A_test = ((A_test_raw - am) @ aW) / asd
    np.savez_compressed(CACHE, Z_ref=Z_ref, Z_null=Z_null, Z_pool=Z_pool, vlab_pool=vlab_pool, A_pool=A_pool, A_test=A_test,
                        vlab_test=vlab_test, ytest_true=ytest_true, VICTIM_ACC=VICTIM_ACC)
    print(f"victim test acc {VICTIM_ACC:.4f} | embeddings done in {time.time()-t0:.0f}s, cached to {CACHE}")
print(f"victim-vs-truth agreement on test: {(vlab_test == ytest_true).mean():.4f} | pool label histogram: {np.bincount(vlab_pool, minlength=10)}")


# %% [markdown]
# ## Part 1 — defender machinery

# %%
# ===== Cell 3: Fourier, reference whitening, evaluation =====
def _fourier_numpy(pts, labels, nacc, modes):
    import scipy.sparse as sp
    F_ = np.zeros((nacc, len(modes)), dtype=np.complex128)
    for s in range(0, len(pts), CHUNK):
        P, lab = pts[s:s + CHUNK], labels[s:s + CHUNK]
        E = np.exp(-2j * np.pi * (P @ modes.T))
        F_ += sp.csr_matrix((np.ones(len(P)), (lab, np.arange(len(P)))), shape=(nacc, len(P))) @ E
    return F_
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

def make_bank(Zref, h, seed):
    # random Fourier modes in the defender's PCA space; mu and Sigma estimated from the historical reference
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / h / (2 * np.pi); m = M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
        Gt = torch.zeros((m, m), dtype=torch.complex128, device=DEV); mut = torch.zeros(m, dtype=torch.complex128, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            E = torch.complex(torch.cos(th), -torch.sin(th)); Gt += E.T @ E.conj(); mut += E.sum(0)
        G, mu = Gt.cpu().numpy(), mut.cpu().numpy()
    else:
        G = np.zeros((m, m), dtype=np.complex128); mu = np.zeros(m, dtype=np.complex128)
        for s0 in range(0, len(Zref), 4096):
            E = np.exp(-2j * np.pi * (Zref[s0:s0 + 4096] @ modes.T)); G += E.T @ E.conj(); mu += E.sum(0)
    G /= len(Zref); mu /= len(Zref)
    Sig = G - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew, U = np.linalg.eigh(Sig); keep = ew > FLOOR * ew.max()
    return dict(h=h, modes=modes, mu=mu, Wk=U[:, keep].conj() / np.sqrt(ew[keep]), meff=int(keep.sum()))

def _auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1 = len(pos); return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))

def evaluate(pts, lab, cnt, banks, Kc, thr=None):
    A_ = len(cnt); iu = np.triu_indices(A_, 1); pc = (iu[0] < Kc) & (iu[1] < Kc)
    Fall = fourier_accounts(pts, lab, A_, np.vstack([b["modes"] for b in banks])); sq = np.sqrt(cnt); Ncol = cnt[:Kc].sum(); out = []
    for j, b in enumerate(banks):
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]
        Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / Ncol)
        rms2 = (np.abs(Fw) ** 2).mean(1)                       # per-account whitened discrepancy (~1 under null)
        ev_raw, V_raw = np.linalg.eigh(Fw @ Fw.conj().T / b["meff"])        # positive side: NO row normalisation
        Fn = Fw / np.sqrt(rms2)[:, None]
        ev, V = np.linalg.eigh(Fn @ Fn.conj().T / b["meff"])                 # missing mode: row-normalised
        r = dict(lam_min=ev[0], lam_max_raw=ev_raw[-1], acct_max=float(rms2.max()), S_union=S_un, z_oracle=(1 - S_un) * math.sqrt(b["meff"]))
        if thr is not None:
            t = thr[j]; nb_ = int((ev < t["tmin"]).sum()); na_ = int((ev_raw > t["tmax_raw"]).sum())
            r.update(mm=float(nb_ == 1 and na_ == 0), max=float(na_ >= 1), acct=float(rms2.max() > t["tacct"]),
                     oracle_low=float(S_un < t["olow"]), oracle_high=float(S_un > t["ohigh"]))
            v = V_raw[:, -1] if (na_ >= 1 and nb_ == 0) else V[:, 0]
            r["auc_spectral"] = _auc(np.abs(v[:Kc]) ** 2, np.abs(v[Kc:]) ** 2)
            r["auc_acct"] = _auc(rms2[:Kc], rms2[Kc:])            # per-account ranking baseline
            mu_a = Fr / cnt[:, None]; G = mu_a @ mu_a.conj().T; nrm = np.real(np.diag(G))
            D2 = (nrm[:, None] + nrm[None, :] - 2 * np.real(G))[iu]; r["auc_pairwise"] = _auc(-D2[pc], -D2[~pc])
        out.append(r)
    return out

def world(campZ, A_, rng):
    Kc = A_ // 5; camp = campZ[rng.permutation(len(campZ))]; lab_c = rng.integers(0, Kc, len(camp)); n_per = len(camp) // Kc
    ben = Z_ref[rng.integers(0, len(Z_ref), (A_ - Kc) * n_per)]; lab_b = Kc + np.repeat(np.arange(A_ - Kc), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b]); return pts, lab, np.bincount(lab, minlength=A_), Kc

t0 = time.time(); BANKS = [make_bank(Z_ref, h, 777 + int(10 * h)) for h in HS]
print("banks: " + ", ".join(f"h={b['h']}: m_eff={b['meff']}" for b in BANKS) + f"  ({time.time()-t0:.0f}s)")
RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, EPOCHS=EPOCHS, N_REF=N_REF, POOL=POOL, CHECKS=CHECKS, TRIALS=TRIALS, AS=AS,
                          N_NULL=N_NULL, W=W, HS=HS, backend="gpu" if USE_GPU else "cpu"),
           "victim_acc": float(VICTIM_ACC), "banks": [dict(h=b["h"], meff=b["meff"]) for b in BANKS]}

# %% [markdown]
# ## Part 2 — extraction campaigns

# %%
# ===== Cell 4: attacker strategies =====
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
def fit_sub(idx):
    clf = LogisticRegression(max_iter=300, C=1.0); clf.fit(A_pool[idx], vlab_pool[idx]); return clf
def fidelity(clf): return float((clf.predict(A_test) == vlab_test).mean())
def proba(clf, X):
    P = np.zeros((len(X), 10)); P[:, clf.classes_] = clf.predict_proba(X); return P
def kcenter_pick(X, sel, k):
    dmin = np.full(len(X), np.inf); S = X[sel]; xn = (X ** 2).sum(1)
    for s0 in range(0, len(X), 4096):
        Xi = X[s0:s0 + 4096]; d2 = xn[s0:s0 + 4096, None] + (S ** 2).sum(1)[None, :] - 2 * Xi @ S.T; dmin[s0:s0 + 4096] = d2.min(1)
    dmin[sel] = -1; pick = []
    for _ in range(k):
        i = int(np.argmax(dmin)); pick.append(i); dmin = np.minimum(dmin, ((X - X[i]) ** 2).sum(1)); dmin[i] = -1
    return np.array(pick)
def _herd_numpy(Phi, N):
    base = Phi @ Phi.mean(0); Ps = np.zeros(len(Phi)); taken = np.zeros(len(Phi), bool); sel = np.empty(N, int)
    for t in range(N):
        sc = base - Ps / (t + 1); sc[taken] = -np.inf; i = int(np.argmax(sc)); sel[t] = i; taken[i] = True; Ps += Phi @ Phi[i]
    return sel
def _herd_torch(Phi, N):
    P = torch.as_tensor(Phi, dtype=torch.float32, device=DEV); base = P @ P.mean(0)
    Ps = torch.zeros(P.shape[0], dtype=torch.float32, device=DEV); taken = torch.zeros(P.shape[0], dtype=torch.bool, device=DEV)
    sel = torch.empty(N, dtype=torch.long, device=DEV)
    for t in range(N):
        sc = (base - Ps / (t + 1)).masked_fill(taken, float("-inf")); i = torch.argmax(sc); sel[t] = i; taken[i] = True; Ps += P @ P[i]
    return sel.cpu().numpy()
def herd_order(X, N, rng, D=256):
    sub = X[rng.choice(len(X), 2000, replace=False)]; d = np.sqrt(((sub[:1000] - sub[1000:]) ** 2).sum(1)); h = np.median(d)
    W_ = rng.standard_normal((X.shape[1], D)) / h; b = rng.uniform(0, 2 * np.pi, D); Phi = np.sqrt(2.0 / D) * np.cos(X @ W_ + b)
    return _herd_torch(Phi, N) if USE_GPU else _herd_numpy(Phi, N)

def run_strategy(name, trial):
    rng = np.random.default_rng(SEED + 1000 * trial + STRATS.index(name)); P = len(A_pool); Nmax = max(TARGETS); fid = {}
    if name in ("random", "herding"):
        order = rng.permutation(P)[:Nmax] if name == "random" else herd_order(A_pool, Nmax, rng)
        for N in CHECKS: fid[N] = fidelity(fit_sub(order[:N]))
        return order, fid
    sel = list(rng.choice(P, SEED_N, replace=False)); taken = np.zeros(P, bool); taken[sel] = True; clf = fit_sub(np.array(sel))
    for tgt in TARGETS:
        k = tgt - len(sel); cand = np.where(~taken)[0]
        if name == "entropy":
            pr = proba(clf, A_pool[cand]); ent = -(pr * np.log(pr + 1e-12)).sum(1); pick = cand[np.argsort(-ent)[:k]]
        elif name == "kcenter":
            pick = kcenter_pick(proba(clf, A_pool), np.array(sel), k)
        else:  # class_balanced
            pred = clf.predict(A_pool[cand]); pick = []
            for c in range(10):
                pool_c = cand[pred == c]; pick += list(rng.choice(pool_c, min(len(pool_c), k // 10), replace=False))
            rest = np.setdiff1d(cand, pick); pick = np.array(pick + list(rng.choice(rest, k - len(pick), replace=False)))
        sel += list(pick); taken[pick] = True; clf = fit_sub(np.array(sel))
        if tgt in CHECKS: fid[tgt] = fidelity(clf)
    return np.array(sel), fid

t0 = time.time(); CAMPAIGNS = {}
for name in STRATS:
    for tr_ in range(TRIALS):
        order, fid = run_strategy(name, tr_); CAMPAIGNS[(name, tr_)] = (order, fid)
        print(f"{name:>15} trial {tr_}: fidelity " + " ".join(f"N={N}:{fid[N]:.4f}" for N in CHECKS) + f"  ({time.time()-t0:.0f}s)")
FID = {(n, N): (float(np.mean([CAMPAIGNS[(n, t)][1][N] for t in range(TRIALS)])), float(np.std([CAMPAIGNS[(n, t)][1][N] for t in range(TRIALS)])))
       for n in STRATS for N in CHECKS}
print("\nmean fidelity (agreement with victim on test):")
for n in STRATS: print(f"{n:>15}: " + "  ".join(f"N={N}: {FID[(n, N)][0]:.4f}±{FID[(n, N)][1]:.4f} (Δ {100*(FID[(n, N)][0]-FID[('random', N)][0]):+.2f} pt)" for N in CHECKS))
RESULTS["fidelity"] = {f"{n}|{N}": FID[(n, N)] for n in STRATS for N in CHECKS}

# %%
# ===== Cell 5: Step-6 evaluation (equal-size colluders, global stream, Higher Criticism, Bonferroni over bandwidths) =====
ALPHA_B = ALPHA / len(HS)
def world(campZ, A_, K_, rng):
    n_per = len(campZ) // K_; camp = campZ[rng.permutation(len(campZ))][:K_ * n_per]; lab_c = np.arange(K_ * n_per) % K_
    ben = Z_ref[rng.integers(0, len(Z_ref), (A_ - K_) * n_per)]; lab_b = K_ + np.repeat(np.arange(A_ - K_), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b]); return pts, lab, np.bincount(lab, minlength=A_)

def stats(pts, lab, cnt, Kc, vectors=False):
    A_ = len(cnt); Fall = fourier_accounts(pts, lab, A_, np.vstack([b["modes"] for b in BANKS])); sq = np.sqrt(cnt)
    out = []
    for j, b in enumerate(BANKS):
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]; Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        rms2 = (np.abs(Fw) ** 2).mean(1); Fn = Fw / np.sqrt(rms2)[:, None]
        Craw, Cn = Fw @ Fw.conj().T / b["meff"], Fn @ Fn.conj().T / b["meff"]
        if vectors: evr, Vr = np.linalg.eigh(Craw); evn = np.linalg.eigvalsh(Cn)
        else: evr, Vr, evn = np.linalg.eigvalsh(Craw), None, np.linalg.eigvalsh(Cn)
        S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / cnt[:Kc].sum())
        S_all = float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum())
        out.append(dict(rms2=rms2, evr=evr, evn=evn, Vr=Vr, S_union=S_un, S_all=S_all, mm_ok=A_ <= 0.8 * b["meff"]))
    return out

def hc_stat(rms2, null_sorted):
    M_ = len(null_sorted); p = (1 + M_ - np.searchsorted(null_sorted, rms2, side="left")) / (M_ + 1)
    p = np.sort(p); A_ = len(p); k = A_ // 2; i = np.arange(1, k + 1); pp = p[:k]
    return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))
DETECTORS = ["spec", "glob_hi", "hc", "acct", "mm", "glob_lo"]
print("Step-6 evaluation ready; Bonferroni alpha per bandwidth =", round(ALPHA_B, 5))

# %% [markdown]
# ## Part 3 — null calibration

# %%
# ===== Part 3: null calibration per (A, K, N) =====
t0 = time.time(); NULLC = {}
q = lambda a, p: float(np.quantile(a, p))
for (A_, K_) in AK:
    for N in CHECKS:
        rec = [dict(mn=[], mx=[], ac=[], ga=[], r2=[]) for _ in BANKS]
        for t in range(N_NULL):
            rng = np.random.default_rng(SEED + 11_000_000 + 7919 * t + 101 * A_ + 13 * K_ + N)
            pts, lab, cnt = world(Z_null[rng.choice(len(Z_null), N, replace=False)], A_, K_, rng)
            for j, s in enumerate(stats(pts, lab, cnt, K_)):
                rec[j]["mn"].append(s["evn"][0]); rec[j]["mx"].append(s["evr"][-1]); rec[j]["ac"].append(s["rms2"].max())
                rec[j]["ga"].append(s["S_all"]); rec[j]["r2"].append(s["rms2"])
        thr = []
        for j, r in enumerate(rec):
            ns = np.sort(np.concatenate(r["r2"])); hcs = [hc_stat(x_, ns) for x_ in r["r2"]]
            thr.append(dict(tmin=q(r["mn"], ALPHA_B), tmax=q(r["mx"], 1 - ALPHA_B), tacct=q(r["ac"], 1 - ALPHA_B), thc=q(hcs, 1 - ALPHA_B),
                            tglo=q(r["ga"], ALPHA_B), tghi=q(r["ga"], 1 - ALPHA_B), null=ns))
        NULLC[(A_, K_, N)] = thr
        print(f"A={A_:3d} K={K_:2d} N={N:5d}: calibrated ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part 4 — detection

# %%
# ===== Part 4: detection with all detectors =====
t0 = time.time(); rows6 = []
for name in STRATS:
    for N in CHECKS:
        for (A_, K_) in AK:
            thr = NULLC[(A_, K_, N)]; fires = {d: 0 for d in DETECTORS}; aucs, aucs_a, Sun = [], [], []; n = 0
            for tr_ in range(TRIALS):
                campZ = Z_pool[CAMPAIGNS[(name, tr_)][0][:N]]
                for w in range(W):
                    rng = np.random.default_rng(SEED + 13_000_000 + 1000 * tr_ + w + 7 * N + 131 * A_ + 17 * K_)
                    pts, lab, cnt = world(campZ, A_, K_, rng); st = stats(pts, lab, cnt, K_, vectors=True); n += 1
                    f = {d: False for d in DETECTORS}; best = None
                    for s, t in zip(st, thr):
                        nb_ = int((s["evn"] < t["tmin"]).sum()) if s["mm_ok"] else 0; na_ = int((s["evr"] > t["tmax"]).sum())
                        f["spec"] |= na_ >= 1; f["mm"] |= (nb_ == 1 and na_ == 0); f["acct"] |= bool(s["rms2"].max() > t["tacct"])
                        f["hc"] |= hc_stat(s["rms2"], t["null"]) > t["thc"]; f["glob_hi"] |= s["S_all"] > t["tghi"]; f["glob_lo"] |= s["S_all"] < t["tglo"]
                        if best is None or s["evr"][-1] / t["tmax"] > best[0]: best = (s["evr"][-1] / t["tmax"], s)
                    for d in DETECTORS: fires[d] += int(f[d])
                    s = best[1]; v = s["Vr"][:, -1]
                    aucs.append(_auc(np.abs(v[:K_]) ** 2, np.abs(v[K_:]) ** 2)); aucs_a.append(_auc(s["rms2"][:K_], s["rms2"][K_:]))
                    Sun.append(np.mean([x_["S_union"] for x_ in st]))
            row = dict(strategy=name, N=N, A=A_, K=K_, f=K_ / A_, n_per=N // K_, n=n, S_union=float(np.mean(Sun)),
                       auc_spec=float(np.mean(aucs)), auc_acct=float(np.mean(aucs_a)))
            for d in DETECTORS: row[d] = fires[d] / n; row[d + "_ci"] = wilson(fires[d], n)
            rows6.append(row)
            print(f"{name:>8} N={N:5d} A={A_:3d} K={K_:2d} (n/acct {N//K_:4d}) S={row['S_union']:6.2f} | " +
                  " ".join(f"{d}={row[d]:.2f}" for d in DETECTORS) + f" | AUC spec/acct {row['auc_spec']:.2f}/{row['auc_acct']:.2f}  ({time.time()-t0:.0f}s)")
RESULTS["step6"] = rows6

# %% [markdown]
# ## Part 5 — predictions

# %%
# ===== Part 5: predictions =====
# [PAPER] App. P (Step 6, "Random campaigns ≤ 2%; linkage AUC ≈ 0.5"): Pass (1.9%; 0.49–0.51)
print("P6.3 random-campaign false alarms (nominal 0.01 combined):")
for d in DETECTORS: print(f"   {d:>8}: {np.mean([r[d] for r in rows6 if r['strategy'] == 'random']):.3f}")
print("   spectral linkage AUC on random campaigns by A: " + ", ".join(f"A={a}: {np.mean([r['auc_spec'] for r in rows6 if r['strategy']=='random' and r['A']==a]):.3f}" for a in AS))
print("\nP6.1/P6.2 head-to-head on concentrated strategies (power, 95% CI):")
print(f"{'strategy':>8} {'N':>5} {'A':>4} {'K':>3} {'K/sqrtA':>7} | {'spectral':>17} {'global':>17} {'HC':>17} {'per-acct':>17} | winner")
H2H = []
# [PAPER] App. E 'CIFAR-10 regime map': spectral wins for K ≲ 2.3 √A; per-account/HC with few heavy colluders; global with many (winner column)
for r in rows6:
    if r["strategy"] not in ("entropy", "kcenter", "herding"): continue
    cell = lambda d: f"{r[d]:.2f}[{r[d+'_ci'][0]:.2f},{r[d+'_ci'][1]:.2f}]"
    scores = {d: r[d] for d in ["spec", "glob_hi", "hc", "acct"]}; top = max(scores.values())
    win = "/".join(d for d, v in scores.items() if v >= top - 1e-9) if top > 0.1 else "none"
    H2H.append(dict(strategy=r["strategy"], N=r["N"], A=r["A"], K=r["K"], ratio=r["K"] / math.sqrt(r["A"]), **scores, winner=win))
    print(f"{r['strategy']:>8} {r['N']:5d} {r['A']:4d} {r['K']:3d} {r['K']/math.sqrt(r['A']):7.2f} | {cell('spec'):>17} {cell('glob_hi'):>17} {cell('hc'):>17} {cell('acct'):>17} | {win}")
inf = [h for h in H2H if 0.05 < max(h["spec"], h["glob_hi"]) < 0.999 or abs(h["spec"] - h["glob_hi"]) > 0.05]
lo = [h for h in inf if h["ratio"] < 2.3]; hi = [h for h in inf if h["ratio"] >= 2.3]
# [PAPER] App. P (Step 6, "Spectral vs global crossover at K ≈ 2.3 √A"): Pass (7 of 8 cells; 2/2)
print(f"\nP6.1 informative cells (not both saturated): K<2.3√A -> spectral≥global in {sum(h['spec']>=h['glob_hi'] for h in lo)}/{len(lo)}; "
      f"K≥2.3√A -> global≥spectral in {sum(h['glob_hi']>=h['spec'] for h in hi)}/{len(hi)}")
conc = [r for r in rows6 if r["strategy"] in ("entropy", "kcenter") and r["N"] == min(CHECKS)]
# [PAPER] App. P (Step 6, "HC ≥ per-account; spectral ≥ HC at N = 2,000"): Pass (29/30; 8/10)
print(f"P6.2 HC≥acct in {sum(r['hc']>=r['acct'] for r in rows6 if r['strategy']!='random')}/{sum(1 for r in rows6 if r['strategy']!='random')} cells; "
      f"spectral≥HC (kcenter/entropy, N={min(CHECKS)}) in {sum(r['spec']>=r['hc'] for r in conc)}/{len(conc)}")
# [PAPER] App. P (Step 6, "No real strategy triggers the lower edge"): Pass (0.002); App. E 'Spectral detection condition'
print(f"P6.4 missing-mode rate on real strategies: {np.mean([r['mm'] for r in rows6 if r['strategy']!='random']):.3f} (nominal ≤ 0.01)")
RESULTS["h2h"] = H2H

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 5)); dc = dict(spec="tab:red", glob_hi="tab:blue", hc="tab:green", acct="tab:gray")
lab = dict(spec="spectral λ_max", glob_hi="global stream", hc="Higher Criticism", acct="per-account max")
for k, (strat, N) in enumerate([("kcenter", min(CHECKS)), ("entropy", min(CHECKS))]):
    rs = [r for r in rows6 if r["strategy"] == strat and r["N"] == N]; xs = np.arange(len(rs))
    for i, d in enumerate(dc):
        y = [r[d] for r in rs]; e = np.array([[r[d] - r[d + "_ci"][0], r[d + "_ci"][1] - r[d]] for r in rs]).T
        ax[k].bar(xs + (i - 1.5) * 0.2, y, 0.2, yerr=e, color=dc[d], label=lab[d], capsize=2)
    ax[k].set_xticks(xs); ax[k].set_xticklabels([f"A={r['A']},K={r['K']}\nK/√A={r['K']/math.sqrt(r['A']):.1f}" for r in rs], fontsize=8)
    ax[k].set_ylabel("power"); ax[k].set_title(f"{strat}, N={N}"); ax[k].legend(fontsize=7)
for h in H2H:
    ax[2].scatter(h["ratio"], h["spec"] - h["glob_hi"], c={"entropy": "tab:orange", "kcenter": "tab:green", "herding": "tab:purple"}[h["strategy"]], s=25)
ax[2].axvline(2.3, color="k", ls=":", label="predicted crossover K=2.3√A"); ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xlabel("K / √A"); ax[2].set_ylabel("power(spectral) − power(global)"); ax[2].set_title("P6.1: when does cross-account structure help?"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step6_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy into the chat

# %%
print("=" * 84); print("STEP 6 VERDICT REPORT"); print("=" * 84)
print("config:", json.dumps(dict(QUICK=QUICK, AK=AK, CHECKS=CHECKS, TRIALS=TRIALS, W=W, N_NULL=N_NULL, HS=HS)), "| victim acc", round(float(VICTIM_ACC), 4))
for n_ in STRATS: print(f"[fid] {n_:>8}: " + " ".join(f"N={N}:{FID[(n_, N)][0]:.4f}" for N in CHECKS))
for r in rows6:
    print(f"[d] {r['strategy']:>8} N={r['N']:5d} A={r['A']:3d} K={r['K']:2d} S={r['S_union']:.2f} " + " ".join(f"{d}={r[d]:.2f}" for d in DETECTORS)
          + f" AUC={r['auc_spec']:.2f}/{r['auc_acct']:.2f}")
print("=" * 84)
with open(f"{OUTDIR}/step6_results.json", "w") as fh: json.dump({"rows": rows6, "h2h": RESULTS["h2h"]}, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step6_results.json and step6_figure.png")
