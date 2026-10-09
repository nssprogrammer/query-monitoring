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
# | App. Table 12, row "MetaMathQA, one prompt per source problem", cols "Two-scale" / "Global" / "MMD" | paired detection at 1% (weekly, K = 50, 1,641 queries) | 0.85 [0.74, 0.94] / 0.65 / 0.56 | Cell 9 `run_block(PART_M, SRC_M, "weekly")`, source `metamath_unique` (`twoscale`, `glob`, `mmd`); Cell 11 "[M3]" lines |
# | App. Table 12, row "Half natural prompts mixed into every account" | paired detection: two-scale / global / MMD | 0.21 [0.10, 0.37] / 0.12 / 0.08 | Cell 9, source `metamath_pad50`; Cell 11 "[M3]" lines |
# | App. Table 12, row "Each prompt followed by a real follow-up turn" | paired detection: two-scale / global / MMD | 0.24 [0.11, 0.38] / 0.12 / 0.08 | Cell 9, source `metamath_followup`; Cell 11 "[M3]" lines |
# | App. Table 12, row "Natural WildChat cluster" | paired detection: two-scale / global / MMD | 0.01 / 0.01 / 0.07 | Cell 9, source `wild_cluster`; Cell 11 "[M3]" lines |
# | App. Table 12, caption | base Qwen2.5-0.5B GSM8K accuracy (first 500 test problems) | 34.4% | Cell 10, `ACC[("base", 0)]` |
# | App. Table 12, col "GSM8K gain, 1,641 queries" | gain: one prompt per problem / half natural / follow-up | +11.4 / +12.0 / +13.4 | Cell 10 loop over `CONFIGS` (budget `BUDGETS[0]`); Cell 11 "[D]" lines |
# | App. Table 12, col "GSM8K gain, 7,550 queries" | gain: one prompt per problem / half natural / natural cluster | +12.6 / +11.0 / +6.0 | Cell 10 loop over `CONFIGS` (budget `BUDGETS[1]`); Cell 11 "[D]" lines |
# | §5 "Padding, and a fix that looks at directions"; App. D.4 | two-scale detection cut by padding / follow-up mimicry | from 0.85 to 0.21–0.24 | Cell 9 (`metamath_unique` vs `metamath_pad50`, `metamath_followup`) |
# | App. P "Corrections made during the study" ("Padding construction") | two-scale detection with padding mixed into every account | 0.21 when mixed | Cell 9, source `metamath_pad50` (construction: Cell 7 `campaign_points`) |
# | Table 2, Table 15 (row "Half padding", two-scale, development weeks); Table 3 caption | two-scale weekly detection of half padding, development weeks | 0.21 | plausibly Cell 9, source `metamath_pad50` (the same value is also logged in Steps 18 and 22) |
# | App. D ("A note on uncertainty"); Table 2 / Table 15 ("0.79–0.85") | one of four runs of two-scale detection of MetaMathQA campaigns | 0.85 | plausibly Cell 9, source `metamath_unique` |
# | App. F "Mechanism probes" (text); App. P (Step 14), "Removing follow-ups closes ≥ 50% of the clumping gap" | P14.1 (fine-scale clumping, all turns vs first turns) | Fail (moved the wrong way) | Cell 8 (`M1`); Cell 11 "P14.1" print |
# | App. P (Step 14), "Follow-up mimicry cuts two-scale power by ≥ 0.3" | P14.2 | Pass (0.85 to 0.24) | Cell 11 "P14.2" print |
# | App. P (Step 14), "Unique at 7,550 gains ≥ 5 points; natural ≤ 1" | P14.3 | Partial (+12.6; natural +6.0) | Cell 11 "P14.3" print |
# | App. P (Step 14), "Evading variants gain ≤ 60% of unique" | P14.4 (ratio of gains at 1,641 queries) | Fail (105% and 118%) | Cell 11 "P14.4" print (ratio) |
#
# Not reported in the paper: the numeric M1 clumping values and M2 two-scale null scores (the paper states only the direction); the `spec`, `hc`, `acct`, `fraudar` and `vol` detectors and the held-out FPR line; the `random` source; two-scale group precision/recall; the per-cell "standard error about 2.2 points" of App. Table 12 is not computed here; `step14_figure.png`.

# %% [markdown]
# # Step 14 — Why the two-scale test works (conversation hypothesis) and what detected vs evading campaigns actually extract
# **Kaggle:** GPU (T4/P100) + Internet + `HF_TOKEN` → **Run All** (≈90–120 min: ≈5 min to re-read WildChat metadata, ≈20 min detection, ≈60–80 min for 6 LoRA fine-tunes and GSM8K evaluations). `QUICK=True` first (skips nothing, tiny sizes).
#
# **Part M — mechanism.** Step 13 falsified paraphrase-level deduplication. Hypothesis: benign groups are clumpy at fine scale because of **conversations** (follow-up turns sit near the turns before them); distillation campaigns send independent single-shot prompts.
# * **M1** per-account fine-scale clumping c = log S̄_fine − log S̄_coarse for benign accounts (all turns vs first turns only) and for colluder accounts.
# * **M2** two-scale null scores of the same benign windows with all turns vs first turns only.
# * **M3** a mimicry attack: `metamath_followup` sends one real WildChat follow-up turn after every MetaMathQA prompt (same account, same total volume). Also `metamath_unique`, `metamath_pad50`, `wild_cluster`, `random`. Weekly populations, K = 50, volume-matched, as in Steps 12–13.
#
# **Part D — distillation.** Student: `Qwen/Qwen2.5-0.5B` (base) + LoRA (r = 16, 1 epoch, lr 2e-4). Teacher answers: MetaMathQA responses (GPT-3.5) for math prompts, WildChat's own ChatGPT replies for natural prompts and follow-ups. Each campaign's training set is exactly the (prompt, answer) pairs it would collect with its query budget. Evaluation: GSM8K test (first 500 problems), greedy, last-number match. One seed per configuration.
#
# | ID | Locked prediction |
# |---|---|
# | **P14.1** | Benign accounts are clumpier than colluder accounts (`metamath_unique`), and removing follow-up turns closes ≥ 50% of that gap |
# | **P14.2** | Follow-up mimicry cuts two-scale power on MetaMathQA by ≥ 0.3 (weekly, K = 50) |
# | **P14.3** | `metamath_unique` with ≈7,550 queries raises GSM8K accuracy by ≥ 5 points over the base model; `wild_cluster` at the same budget by ≤ 1 point |
# | **P14.4** | Price of stealth: at ≈1,640 queries, the evading variants (`metamath_pad50`, `metamath_followup`) gain ≤ 60% of `metamath_unique`'s GSM8K improvement |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc, re
import numpy as np
QUICK = False
SYNTHETIC_TEST = False
SEED = 9091
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
H_COARSE, H_FINE, M_MODES, ALPHA, FLOOR = 4.0, 0.5, 300, 0.01, 1e-4
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, SUB_FRAC = 500, 10, 0.2, 0.7
N_TOPICS, N_REF_BATCH, REF_BATCH, MMD_WIN = 100, 20, 500, 2000
DETS = ["spec", "hc", "acct", "glob", "mmd", "fraudar", "vol", "twoscale"]
SRC_M = ["metamath_unique", "metamath_followup", "metamath_pad50", "wild_cluster", "random"]
STUDENT = "Qwen/Qwen2.5-0.5B"
if QUICK:
    PART_M = dict(BLOCK=1, KS=[20], CAL_N=4, LAGB=2, N_TEST=3, NULL_PER=12, REPS=2)
    MMD_WIN, N_REF_BATCH, REF_BATCH = 400, 5, 200
    N_DIAG_WEEKS, BUDGETS, N_EVAL, MAX_LEN, BS, GEN_BS, MAX_NEW = 2, [40, 80], 16, 256, 4, 8, 64
else:
    PART_M = dict(BLOCK=1, KS=[50], CAL_N=8, LAGB=4, N_TEST=24, NULL_PER=40, REPS=4)
    N_DIAG_WEEKS, BUDGETS, N_EVAL, MAX_LEN, BS, GEN_BS, MAX_NEW = 8, [1641, 7550], 500, 512, 8, 32, 320
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE14 = f"{OUTDIR}/step14_meta_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
N_CONV = 30000 if QUICK else None; DIMS = [384]
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} | student {STUDENT}")

# %% [markdown]
# ## Part 0 — WildChat embeddings (Step 11 cache), conversation metadata, MetaMathQA pairs

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


# %%
# ===== Cell 3: conversation ids, turn indices and attacker-pool text pairs (same order as the cache), MetaMathQA pairs =====
import pickle
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC    # identical to Steps 12-13
PAIRS_PKL = CACHE14.replace(".npz", "_pairs.pkl")
if os.path.exists(CACHE14) and os.path.exists(PAIRS_PKL):
    _c = np.load(CACHE14); CONV, TURN, D_MMU = _c["CONV"], _c["TURN"], _c["D_MMU"]
    with open(PAIRS_PKL, "rb") as fh: POOL_TXT, MM_PAIRS = pickle.load(fh)
    print(f"loaded metadata cache: {len(CONV)} turns, {len(POOL_TXT)} pool pairs, {len(MM_PAIRS)} MetaMathQA pairs")
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        r = np.random.default_rng(5); CONV = np.cumsum(r.random(len(E16)) < 0.4).astype(np.int32); TURN = np.ones(len(E16), np.int16)
        for i in range(1, len(E16)):
            if CONV[i] == CONV[i - 1]: TURN[i] = TURN[i - 1] + 1
        POOL_TXT = {int(i): (f"user prompt {i}", f"assistant reply {i} " * 5) for i in np.flatnonzero(is_pool[ACCT])}
        MM_PAIRS = [(f"What is {a} plus {a+1}?", f"{a} + {a+1} = {2*a+1}. The answer is: {2*a+1}") for a in range(1500)]
        D_MMU = (np.random.default_rng(3).standard_normal(384) * 2 + np.random.default_rng(4).standard_normal((1500, 384)) * 0.7).astype(np.float16)
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        split = "train" if N_CONV is None else f"train[:{N_CONV}]"
        ds = load_dataset("allenai/WildChat-1M", split=split, token=tok).select_columns(["conversation", "hashed_ip", "timestamp", "language"])
        ips, conv_id, turn_ix, users, replies = [], [], [], [], []; cid = 0
        for batch in ds.iter(batch_size=2000):
            for conv, ip, ts, lang in zip(batch["conversation"], batch["hashed_ip"], batch["timestamp"], batch["language"]):
                if lang != "English" or ip is None or ts is None: continue
                t = 0
                for j, turn in enumerate(conv):
                    if turn.get("role") == "user" and turn.get("content"):
                        t += 1; nxt = next((c.get("content") or "" for c in conv[j + 1:] if c.get("role") == "assistant"), "")
                        ips.append(ip); conv_id.append(cid); turn_ix.append(t); users.append(turn["content"][:1000]); replies.append(nxt[:1500])
                cid += 1
        _, ACCT2 = np.unique(np.array(ips), return_inverse=True)
        assert len(ACCT2) == len(ACCT) and np.array_equal(ACCT2.astype(np.int32), ACCT), "re-read order does not match the Step 11 cache"
        CONV, TURN = np.array(conv_id, np.int32), np.array(turn_ix, np.int16)
        POOL_TXT = {int(i): (users[i], replies[i]) for i in np.flatnonzero(is_pool[ACCT])}
        del users, replies, ds; gc.collect()
        print(f"re-read WildChat: order verified, {len(CONV)} turns in {int(CONV.max())+1} conversations, {len(POOL_TXT)} pool pairs ({time.time()-t0:.0f}s)")
        mm = load_dataset("meta-math/MetaMathQA", split="train", token=tok).select_columns(["query", "response", "original_question"])
        groups = {}
        for q, a, o in zip(mm["query"], mm["response"], mm["original_question"]):
            if q and a and o: groups.setdefault(o, []).append((q.strip()[:1000], a.strip()[:2000]))
        rr = np.random.default_rng(SEED); origs = list(groups); rr.shuffle(origs)
        MM_PAIRS = [groups[o][int(rr.integers(len(groups[o])))] for o in origs]          # one (query, response) per source problem
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        D_MMU = enc.encode([q for q, _ in MM_PAIRS], batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        print(f"MetaMathQA: {len(MM_PAIRS)} (query, response) pairs, one per source problem ({time.time()-t0:.0f}s)")
    np.savez(CACHE14, CONV=CONV, TURN=TURN, D_MMU=D_MMU)
    with open(PAIRS_PKL, "wb") as fh: pickle.dump((POOL_TXT, MM_PAIRS), fh)
DIST = {"metamath_unique": D_MMU}
print(f"follow-up turns (turn >= 2) among all turns: {np.mean(TURN >= 2):.3f}")

# %% [markdown]
# ## Part 1 — projection, topics, natural source cluster; statistics; pipeline

# %%
# ===== Cell 4: projection, topics, natural source cluster with farthest-point order =====
from sklearn.cluster import MiniBatchKMeans
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC; pool_idx = np.flatnonzero(is_pool[ACCT])
Z = make_Z(384); W384 = UB[:, :384] / math.sqrt(EW[:384].sum() / 4.0); EN = E16
DZ = {k: ((v.astype(np.float32) - MU_E) @ W384).astype(np.float32) for k, v in DIST.items()}
t0 = time.time(); ben = np.flatnonzero(~is_pool[ACCT])
km = MiniBatchKMeans(n_clusters=N_TOPICS, random_state=SEED, batch_size=4096, n_init=3).fit(Z[np.random.default_rng(1).choice(ben, min(200000, len(ben)), replace=False)])
TOPIC = km.predict(Z).astype(np.int32)
CL10 = MiniBatchKMeans(n_clusters=10, random_state=SEED + 1, batch_size=4096, n_init=3).fit(Z[pool_idx]).labels_
WILD = pool_idx[CL10 == int(np.argmax(np.bincount(CL10)))]
def fps(X, N, seed=0):
    N = min(N, len(X)); r = np.random.default_rng(seed); i0 = int(r.integers(len(X)))
    if USE_GPU:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV); d2 = ((Xt - Xt[i0]) ** 2).sum(1); sel = torch.empty(N, dtype=torch.long, device=DEV); sel[0] = i0
        for t in range(1, N): i = torch.argmax(d2); sel[t] = i; d2 = torch.minimum(d2, ((Xt - Xt[i]) ** 2).sum(1))
        return sel.cpu().numpy()
    sel = [i0]; d2 = ((X - X[i0]) ** 2).sum(1)
    for _ in range(N - 1): i = int(np.argmax(d2)); sel.append(i); d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.array(sel)
WILD_FPS = WILD[fps(Z[WILD], min(len(WILD), 20000), seed=SEED)]
_sub = EN[np.random.default_rng(2).choice(len(EN), 2000, replace=False)].astype(np.float32)
_d2 = ((_sub[:1000, None, :] - _sub[None, 1000:, :]) ** 2).sum(-1); MMD_BW = float(np.sqrt(np.median(_d2) / 2)); del _d2
print(f"natural source cluster {len(WILD)} prompts (farthest-point order {len(WILD_FPS)}); sources: " + ", ".join(f"{k}:{len(v)}" for k, v in DIST.items()) + f" ({time.time()-t0:.0f}s)")


# %%
# ===== Cell 5: augmented real whitening, statistics, MMD, Fraudar =====
def make_bank_aug(Zref, h, seed):
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / h / (2 * np.pi); m2 = 2 * M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV); G = torch.zeros((m2, m2), dtype=torch.float64, device=DEV); mu = torch.zeros(m2, dtype=torch.float64, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            X = torch.cat([torch.cos(th), torch.sin(th)], 1); G += X.T @ X; mu += X.sum(0)
        G, mu = G.cpu().numpy(), mu.cpu().numpy()
    else:
        G = np.zeros((m2, m2)); mu = np.zeros(m2)
        for s0 in range(0, len(Zref), 4096):
            th = 2 * np.pi * (Zref[s0:s0 + 4096] @ modes.T); X = np.hstack([np.cos(th), np.sin(th)]); G += X.T @ X; mu += X.sum(0)
    G /= len(Zref); mu /= len(Zref); Sig = G - np.outer(mu, mu); ew, U = np.linalg.eigh((Sig + Sig.T) / 2); keep = ew > FLOOR * ew.max()
    B = dict(modes=modes, mu=mu, W=U[:, keep] / np.sqrt(ew[keep]), r=int(keep.sum()))
    if USE_GPU:
        B["MT"] = torch.as_tensor(modes, dtype=torch.float64, device=DEV); B["MU_T"] = torch.as_tensor(mu, dtype=torch.float64, device=DEV); B["W_T"] = torch.as_tensor(B["W"], dtype=torch.float64, device=DEV)
    return B
def whitened(pts, lab, cnt, B):
    if USE_GPU:
        Fc = torch.zeros((len(cnt), M_MODES), dtype=torch.float64, device=DEV); Fs = torch.zeros_like(Fc)
        for s0 in range(0, len(pts), CHUNK):
            P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV); ix = torch.as_tensor(lab[s0:s0 + CHUNK], dtype=torch.long, device=DEV)
            th = (2.0 * math.pi) * (P @ B["MT"].T); Fc.index_add_(0, ix, torch.cos(th)); Fs.index_add_(0, ix, torch.sin(th))
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV)
        return (((torch.cat([Fc, Fs], 1) - c[:, None] * B["MU_T"][None, :]) / torch.sqrt(c)[:, None]) @ B["W_T"]).cpu().numpy()
    import scipy.sparse as sp
    X = np.zeros((len(cnt), 2 * M_MODES))
    for s0 in range(0, len(pts), CHUNK):
        P, l_ = pts[s0:s0 + CHUNK], lab[s0:s0 + CHUNK]; th = 2 * np.pi * (P @ B["modes"].T)
        X += sp.csr_matrix((np.ones(len(P)), (l_, np.arange(len(P)))), shape=(len(cnt), len(P))) @ np.hstack([np.cos(th), np.sin(th)])
    return ((X - cnt[:, None] * B["mu"][None, :]) / np.sqrt(cnt)[:, None]) @ B["W"]
def union_S(Y, cnt, ix): return float(((Y[ix] * np.sqrt(cnt[ix])[:, None]).sum(0) ** 2).mean() / cnt[ix].sum())
def mmd2(Xw, refs):
    # Liu-Guo-Dong style: unbiased MMD^2 between window sample and each benign reference batch, averaged
    def k(A_, B_):
        if USE_GPU: return torch.exp(-torch.cdist(A_, B_) ** 2 / (2 * MMD_BW ** 2))
        d2 = (A_ ** 2).sum(1)[:, None] + (B_ ** 2).sum(1)[None, :] - 2 * A_ @ B_.T; return np.exp(-np.maximum(d2, 0) / (2 * MMD_BW ** 2))
    X = torch.as_tensor(Xw, dtype=torch.float32, device=DEV) if USE_GPU else Xw
    n = len(X); Kxx = k(X, X); sxx = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1)); out = []
    for R in refs:
        m = len(R); Kyy = k(R, R); Kxy = k(X, R)
        out.append(float(sxx + (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1)) - 2 * Kxy.mean()))
    return float(np.mean(out))
def fraudar(lab, top, A_):
    C = np.zeros((A_, N_TOPICS)); np.add.at(C, (lab, top), 1.0); wt = 1.0 / np.log(C.sum(0) + 5.0); Wm = C * wt[None, :]
    da, dt = Wm.sum(1), Wm.sum(0); alive_a, alive_t = np.ones(A_, bool), np.ones(N_TOPICS, bool); tot = Wm.sum(); n = A_ + N_TOPICS; best = tot / n
    for _ in range(n - 1):
        ia = np.where(alive_a, da, np.inf).argmin(); it = np.where(alive_t, dt, np.inf).argmin()
        if da[ia] <= dt[it]:
            alive_a[ia] = False; tot -= da[ia]; dt -= Wm[ia] * alive_t; da[ia] = np.inf
        else:
            alive_t[it] = False; tot -= dt[it]; da -= Wm[:, it] * alive_a; dt[it] = np.inf
        n -= 1; best = max(best, tot / max(n, 1))
    return float(best)
def window_stats(idx_pts, lab, cnt, Bc, Bf, refs, extra=None, en_extra=None, top_extra=None):
    pts = Z[idx_pts] if extra is None else np.vstack([extra[0], Z[idx_pts]]); labs = lab if extra is None else np.concatenate([extra[1], lab])
    en = EN[idx_pts] if extra is None else np.vstack([en_extra, EN[idx_pts]]); top = TOPIC[idx_pts] if extra is None else np.concatenate([top_extra, TOPIC[idx_pts]])
    Yc = whitened(pts, labs, cnt, Bc); Yf = whitened(pts, labs, cnt, Bf); A_ = len(cnt)
    ev, V = np.linalg.eigh(Yc.T @ Yc / Bc["r"]); v = Yc @ V[:, -1]; v2 = v ** 2 / (v ** 2).sum()
    o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); grp = o[:m]
    r2 = (Yc ** 2).mean(1); rr = np.random.default_rng(len(labs)); samp = rr.choice(len(en), min(MMD_WIN, len(en)), replace=False)
    return dict(spec=float(ev[-1]), r2=r2, acct=float(r2.max()), glob=union_S(Yc, cnt, np.arange(A_)),
                mmd=mmd2(en[samp].astype(np.float32), refs), fraudar=fraudar(labs, top, A_),
                vol=float(np.sort(cnt)[-10:].sum() / cnt.sum()), twoscale=float(np.log(union_S(Yc, cnt, grp)) - np.log(union_S(Yf, cnt, grp))), v2=v2, grp=grp)
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))


# %%
# ===== Cell 6: the full pipeline for one block length (weekly or 4-week), as a function =====
def campaign_points(src, n, r):
    if src == "random": ix = r.choice(pool_idx, n, replace=False); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster": ix = r.choice(WILD, min(n, len(WILD)), replace=n > len(WILD)); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster_dedup":
        start = int(r.integers(0, max(1, len(WILD_FPS) - n))); ix = WILD_FPS[start:start + n] if n <= len(WILD_FPS) else WILD_FPS
        return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    D = DIST[src]
    if src == "metamath_rephrase":                                     # whole groups of 4 variants of the same source problem
        g = r.permutation(int(MM_GROUP.max()) + 1)[: int(math.ceil(n / 4))]; ix = np.flatnonzero(np.isin(MM_GROUP, g))[:n]
    else:
        ix = r.choice(len(D), min(n, len(D)), replace=n > len(D))
    return DZ[src][ix], D[ix].astype(np.float32), km.predict(DZ[src][ix]).astype(np.int32)
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))
def calibrate(recs, alpha):
    ns = np.sort(np.concatenate([r["r2"] for r in recs])); T = {d: float(np.quantile([r[d] for r in recs], 1 - alpha)) for d in DETS if d != "hc"}
    T["hc"] = float(np.quantile([hc_stat(r["r2"], ns) for r in recs], 1 - alpha)); T["ns"] = ns; return T
def fires(s, T):
    f = {d: s[d] > T[d] for d in DETS if d != "hc"}; f["hc"] = hc_stat(s["r2"], T["ns"]) > T["hc"]; return f
def boot_ci(x, B=1000, rng=np.random.default_rng(7)):
    x = np.array(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

def run_block(P, sources, label, shard=False):
    t0 = time.time(); BLK = P["BLOCK"]; blk = (WEEK // BLK).astype(np.int32); n_b = int(blk.max()) + 1
    order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]
    GA = {b: [] for b in range(n_b)}; rg = np.random.default_rng(SEED + BLK)
    for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
        a = int(ACCT[g[0]]); b = int(blk[g[0]])
        if is_pool[a] or len(g) < MIN_PER_ACCT: continue
        if len(g) > VOL_CAP * BLK: g = rg.choice(g, VOL_CAP * BLK, replace=False)
        GA[b].append(g)
    ok = [b for b in range(P["CAL_N"], n_b) if len(GA[b]) >= 40 and all(len(GA[v]) >= 20 for v in range(b - P["CAL_N"], b))]
    TB = [int(ok[i]) for i in np.linspace(0, len(ok) - 1, min(P["N_TEST"], len(ok))).astype(int)] if ok else []
    print(f"[{label}] accounts per block: " + ", ".join(f"{b}:{len(GA[b])}" for b in range(n_b) if GA[b]) + f"\n[{label}] test blocks {TB}")
    def sub_window(b, r, frac=SUB_FRAC, k=None):
        k = k or max(20, int(frac * len(GA[b]))); pick = r.choice(len(GA[b]), min(k, len(GA[b])), replace=False); parts = [GA[b][i] for i in pick]
        vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
    NEED = sorted(set(v for b in TB for v in range(b - P["CAL_N"], b + 1))); BANK, REFS, NULLW = {}, {}, {}
    for b in NEED:
        prev = [np.concatenate(GA[v]) for v in range(max(0, b - P["LAGB"]), b) if GA[v]]
        ref = np.concatenate(prev) if prev else np.concatenate(GA[b]); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
        BANK[b] = (make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8))
        rb = np.random.default_rng(b + 1).choice(ref, N_REF_BATCH * REF_BATCH, replace=len(ref) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
        REFS[b] = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
        NULLW[b] = []
        for t in range(P["NULL_PER"]):
            idx, lab, vols = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); NULLW[b].append(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]))
    print(f"[{label}] banks and null windows for {len(NEED)} blocks ({time.time()-t0:.0f}s)")
    THR = {b: calibrate([r for v in range(b - P["CAL_N"], b) for r in NULLW[v]], ALPHA) for b in TB}
    FPR = {d: boot_ci([np.mean([fires(r, THR[b])[d] for r in NULLW[b]]) for b in TB]) for d in DETS}
    print(f"[{label}] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS))
    rows = []
    for src in sources:
        for K_ in P["KS"]:
            pw = {d: [] for d in DETS}; prec, rec, Ns = [], [], []
            for b in TB:
                acc = {d: [] for d in DETS}
                for rep in range(P["REPS"]):
                    r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + K_); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
                    if K_ >= A_ // 2: continue
                    base = fires(window_stats(idx, lab, np.bincount(lab, minlength=A_), *BANK[b], REFS[b]), THR[b])
                    cv = r.choice(np.array([len(g) for g in GA[b]]), K_, replace=True); N = int(cv.sum()); Ns.append(N)
                    Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * b + rep + K_ + sources.index(src))); N = len(Zc)
                    labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_
                    _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True); lc, lk = inv[:N], inv[N:]
                    s = window_stats(idx[keep], lk, np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, lc), en_extra=Ec, top_extra=Tc); f = fires(s, THR[b])
                    for d in DETS: acc[d].append(float(f[d] and not base[d]))
                    if f["twoscale"]:
                        g = s["grp"]; nc = int(lc.max()) + 1; prec.append(float(np.mean(g < nc))); rec.append(float(np.sum(g < nc) / nc))
                for d in DETS:
                    if acc[d]: pw[d].append(np.mean(acc[d]))
            row = dict(part=label, source=src, K=K_, N_mean=float(np.mean(Ns)) if Ns else 0.0,
                       ts_precision=float(np.mean(prec)) if prec else float("nan"), ts_recall=float(np.mean(rec)) if rec else float("nan"))
            for d in DETS:
                if pw[d]: m, lo, hi = boot_ci(pw[d]); row[d] = m; row[d + "_ci"] = (lo, hi)
            rows.append(row)
            print(f"[{label}] {src:>19} K={K_:4d} N~{row['N_mean']:7.0f} | " + " ".join(f"{d}={row.get(d, float('nan')):.2f}" for d in DETS)
                  + f" | two-scale group precision {row['ts_precision']:.2f} recall {row['ts_recall']:.2f}  ({time.time()-t0:.0f}s)")
    shard_rows = []
    if shard:
        for src in ["metamath_unique", "random"]:
            per_shard = {d: [] for d in DETS}
            for b in TB:
                nulls = []
                for v in range(b - P["CAL_N"], b):
                    for t in range(P["NULL_PER"]):
                        idx, lab, vols = sub_window(v, np.random.default_rng(SEED + 55 * v + t), k=SHARD); nulls.append(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]))
                Ts = calibrate(nulls, ALPHA); n_sh = max(1, len(GA[b]) // SHARD)
                for rep in range(P["REPS"]):
                    for sh in range(n_sh):
                        r = np.random.default_rng(SEED + 77_000 * b + 100 * rep + sh); idx, lab, vols = sub_window(b, r, k=SHARD)
                        base = fires(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]), Ts)
                        cv = int(r.choice(vols)); Zc, Ec, Tc = campaign_points(src, cv, r); cv = len(Zc); keep = lab >= 1
                        _, inv = np.unique(np.concatenate([np.zeros(cv, int), lab[keep]]), return_inverse=True)
                        s = window_stats(idx[keep], inv[cv:], np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, inv[:cv]), en_extra=Ec, top_extra=Tc); f = fires(s, Ts)
                        for d in DETS: per_shard[d].append(float(f[d] and not base[d]))
            row = dict(part=label + "-shard", source=src, **{d: float(np.mean(v)) for d, v in per_shard.items()}, n=len(per_shard["spec"]))
            shard_rows.append(row)
            print(f"[{label}] sharded (one colluder per {SHARD}-account shard), per-shard paired power at alpha={ALPHA}, {src}: " + " ".join(f"{d}={row[d]:.2f}" for d in DETS) + f" (n={row['n']})")
    return dict(FPR=FPR, rows=rows, shard=shard_rows, TB=TB)


# %%
# ===== Cell 7: campaign sources for Step 14 (overrides Step 13's) =====
FOLLOW = np.array([i for i in POOL_TXT if TURN[i] >= 2])                          # real follow-up turns from attacker-pool accounts
print(f"real follow-up turns available to the attacker: {len(FOLLOW)}")
def campaign_points(src, n, r):
    if src == "random": ix = r.choice(pool_idx, n, replace=False); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster": ix = r.choice(WILD, min(n, len(WILD)), replace=n > len(WILD)); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    D, DZm = DIST["metamath_unique"], DZ["metamath_unique"]
    if src == "metamath_unique":
        ix = r.choice(len(D), min(n, len(D)), replace=n > len(D)); return DZm[ix], D[ix].astype(np.float32), km.predict(DZm[ix]).astype(np.int32)
    nm = (n + 1) // 2; ix = r.choice(len(D), nm, replace=False); other = FOLLOW if src == "metamath_followup" else pool_idx   # even slots: prompts
    jx = r.choice(other, n - nm, replace=False)                                                                            # odd slots: follow-ups / padding
    Zs, Es, Ts = np.empty((n, Z.shape[1]), np.float32), np.empty((n, EN.shape[1]), np.float32), np.empty(n, np.int32)
    Zs[0::2], Es[0::2], Ts[0::2] = DZm[ix], D[ix].astype(np.float32), km.predict(DZm[ix])
    Zs[1::2], Es[1::2], Ts[1::2] = Z[jx], EN[jx].astype(np.float32), TOPIC[jx]
    return Zs, Es, Ts                                                           # prompt, follow-up, prompt, ... so each pair lands in one account


# %% [markdown]
# ## Part M — mechanism

# %%
# ===== Cell 8: Part M1/M2 — fine-scale clumping of benign accounts (all turns vs first turns) and of colluders =====
t0 = time.time(); blk = WEEK.astype(np.int32); n_b = int(blk.max()) + 1
order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]; GA = {b: [] for b in range(n_b)}
for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
    if not is_pool[int(ACCT[g[0]])] and len(g) >= MIN_PER_ACCT: GA[int(blk[g[0]])].append(g[:VOL_CAP])
weeks = [b for b in range(8, n_b) if len(GA[b]) >= 60][:: max(1, len([b for b in range(8, n_b) if len(GA[b]) >= 60]) // N_DIAG_WEEKS)][:N_DIAG_WEEKS]
def clump(pts, lab, cnt, Bc, Bf): return np.log((whitened(pts, lab, cnt, Bf) ** 2).mean(1)) - np.log((whitened(pts, lab, cnt, Bc) ** 2).mean(1))
C_all, C_first, C_col, TS_all, TS_first = [], [], {s: [] for s in ["metamath_unique", "metamath_followup", "wild_cluster", "random"]}, [], []
for b in weeks:
    ref = np.random.default_rng(b).choice(np.concatenate([np.concatenate(GA[v]) for v in range(b - 4, b) if GA[v]]), 100000, replace=True)
    Bc, Bf = make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8)
    accs = [g for g in GA[b] if (TURN[g] == 1).sum() >= 5]
    vols = np.array([len(g) for g in accs]); idx = np.concatenate(accs); lab = np.repeat(np.arange(len(accs)), vols)
    C_all.append(clump(Z[idx], lab, vols, Bc, Bf))
    f_acc = [g[TURN[g] == 1] for g in accs]; fv = np.array([len(g) for g in f_acc]); fidx = np.concatenate(f_acc); flab = np.repeat(np.arange(len(f_acc)), fv)
    C_first.append(clump(Z[fidx], flab, fv, Bc, Bf))
    refs = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in ref[: N_REF_BATCH * REF_BATCH].reshape(N_REF_BATCH, REF_BATCH)]
    for t in range(3):                                                     # M2: two-scale null scores, same accounts, all turns vs first turns
        r = np.random.default_rng(SEED + 31 * b + t); pick = r.choice(len(accs), max(20, int(SUB_FRAC * len(accs))), replace=False)
        for src_acc, store in [(accs, TS_all), (f_acc, TS_first)]:
            parts = [src_acc[i] for i in pick]; v = np.array([len(p) for p in parts]); ii = np.concatenate(parts); ll = np.repeat(np.arange(len(parts)), v)
            store.append(window_stats(ii, ll, v, Bc, Bf, refs)["twoscale"])
    for s in C_col:
        r = np.random.default_rng(SEED + 7 * b); cv = r.choice(np.array([len(g) for g in GA[b]]), 50, replace=True)
        Zc, _, _ = campaign_points(s, int(cv.sum()), r); C_col[s].append(clump(Zc, np.repeat(np.arange(50), cv)[: len(Zc)], cv, Bc, Bf))
    print(f"week {b}: {len(accs)} benign accounts with >=5 first turns ({time.time()-t0:.0f}s)")
med = lambda L: float(np.median(np.concatenate(L)))
# [PAPER] App. P (Step 14, P14.1): removing follow-ups moved benign clumping the wrong way (Fail); App. F "Mechanism probes" (text only)
M1 = dict(benign_all=med(C_all), benign_first=med(C_first), **{f"colluder_{s}": med(v) for s, v in C_col.items()})
M2 = dict(ts_null_all=float(np.median(TS_all)), ts_null_first=float(np.median(TS_first)), ts_null_all_p95=float(np.quantile(TS_all, .95)), ts_null_first_p95=float(np.quantile(TS_first, .95)))
print("M1 median fine-scale clumping c = log S_fine - log S_coarse (higher = clumpier):", {k: round(v, 3) for k, v in M1.items()})
print("M2 two-scale null scores (median, 95th pct): all turns", round(M2["ts_null_all"], 3), round(M2["ts_null_all_p95"], 3), "| first turns only", round(M2["ts_null_first"], 3), round(M2["ts_null_first_p95"], 3))

# %%
# ===== Cell 9: Part M3 — detection with follow-up mimicry (weekly, volume-matched) =====
# [PAPER] App. Table 12 (two-scale / global / MMD at 1,641 queries): 0.85 [0.74, 0.94] / 0.65 / 0.56 unique; 0.21 [0.10, 0.37] / 0.12 / 0.08 pad50;
# [PAPER]   0.24 [0.11, 0.38] / 0.12 / 0.08 follow-up; 0.01 / 0.01 / 0.07 natural cluster
# [PAPER] §5 "Padding, and a fix that looks at directions", App. D.4: two-scale from 0.85 to 0.21–0.24 (plausibly also Table 2/15 dev-week 0.21)
RES_M = run_block(PART_M, SRC_M, "weekly")
DET = {r["source"]: {d: r.get(d) for d in DETS} for r in RES_M["rows"]}

# %% [markdown]
# ## Part D — distillation

# %%
# ===== Cell 10: Part D — LoRA distillation of Qwen2.5-0.5B on what each campaign collects; GSM8K =====
import sys, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])      # old torchao blocks recent peft; LoRA does not need it
for m in [m for m in list(sys.modules) if m == "peft" or m.startswith("peft.") or m == "torchao" or m.startswith("torchao.")]:
    del sys.modules[m]                                                                   # drop half-imported modules from any failed attempt
importlib.invalidate_caches()
try:
    import peft
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"]); importlib.invalidate_caches(); import peft
print("peft", peft.__version__)
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
tokz = AutoTokenizer.from_pretrained(STUDENT); tokz.pad_token = tokz.pad_token or tokz.eos_token
fmtq = lambda q: f"Question: {q.strip()}\nAnswer:"
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N_EVAL))
GOLD = [float(a.split("####")[-1].strip().replace(",", "")) for a in gsm["answer"]]
def last_number(t):
    t = t.split("\nQuestion:")[0]; m = re.search(r"answer is:?\s*\$?(-?[\d,]*\.?\d+)", t)
    if m: s = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", t); s = nums[-1] if nums else None
    try: return float(s.replace(",", "")) if s else None
    except ValueError: return None
@torch.no_grad()
def evaluate(model):
    model.eval(); tokz.padding_side = "left"; correct = 0
    for i in range(0, N_EVAL, GEN_BS):
        qs = gsm["question"][i:i + GEN_BS]; enc = tokz([fmtq(q) for q in qs], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tokz.pad_token_id)
        for j, o in enumerate(out):
            p = last_number(tokz.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            correct += int(p is not None and abs(p - GOLD[i + j]) < 1e-4)
    return correct / N_EVAL
def train(pairs, seed=0):
    torch.manual_seed(seed); model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4); steps = math.ceil(len(pairs) / BS)
    sch = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps); scaler = torch.cuda.amp.GradScaler(enabled=USE_GPU)
    tokz.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs))
    for s0 in range(0, len(pairs), BS):
        batch = [pairs[k] for k in order[s0:s0 + BS]]
        full = [fmtq(q) + " " + a.strip() + tokz.eos_token for q, a in batch]; enc = tokz(full, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tokz(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_GPU):
            loss = model(**enc, labels=labels).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    return model
def campaign_pairs(src, n, r):
    mm = lambda k: [MM_PAIRS[i] for i in r.choice(len(MM_PAIRS), k, replace=False)]
    if src == "metamath_unique": return mm(n)
    if src == "metamath_pad50": return mm(n // 2) + [POOL_TXT[int(i)] for i in r.choice(pool_idx, n - n // 2, replace=False)]
    if src == "metamath_followup": return mm(n // 2) + [POOL_TXT[int(i)] for i in r.choice(FOLLOW, n - n // 2, replace=False)]
    if src == "wild_cluster": return [POOL_TXT[int(i)] for i in r.choice(WILD, n, replace=False)]
CONFIGS = [("base", 0)] + [("metamath_unique", BUDGETS[0]), ("metamath_pad50", BUDGETS[0]), ("metamath_followup", BUDGETS[0]),
                           ("metamath_unique", BUDGETS[1]), ("metamath_pad50", BUDGETS[1]), ("wild_cluster", BUDGETS[1])]
ACC = {}; t0 = time.time()
for src, n in CONFIGS:
    if src == "base": model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.float16 if USE_GPU else torch.float32).to(DEV)
    else: model = train([p for p in campaign_pairs(src, n, np.random.default_rng(SEED + n)) if p[0] and p[1]])
    # [PAPER] App. Table 12: base 34.4%; gains +11.4 / +12.0 / +13.4 at 1,641 queries, +12.6 / +11.0 / +6.0 at 7,550
    ACC[(src, n)] = evaluate(model); del model; gc.collect()
    if USE_GPU: torch.cuda.empty_cache()
    print(f"{src:>18} budget {n:5d}: GSM8K accuracy {ACC[(src, n)]:.3f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 11: predictions, frontier figure, verdict =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
base = ACC[("base", 0)]; gain = {k: v - base for k, v in ACC.items() if k[0] != "base"}; n0, n1 = BUDGETS
gap = M1["benign_all"] - M1["colluder_metamath_unique"]; closed = (M1["benign_all"] - M1["benign_first"]) / gap if gap > 0 else float("nan")
# [PAPER] App. P (Step 14): P14.1 Fail (moved the wrong way)
print(f"P14.1 clumping: benign {M1['benign_all']:.3f} vs colluder {M1['colluder_metamath_unique']:.3f} (gap {gap:.3f}); first turns only {M1['benign_first']:.3f} -> closes {closed:.2f} of the gap (pass if gap > 0 and >= 0.5)")
# [PAPER] App. P (Step 14): P14.2 Pass (0.85 to 0.24)
print(f"P14.2 two-scale power: metamath_unique {fmt(DET['metamath_unique']['twoscale'])} vs metamath_followup {fmt(DET['metamath_followup']['twoscale'])} (pass if drop >= 0.3)")
# [PAPER] App. P (Step 14): P14.3 Partial (+12.6; natural +6.0)
print(f"P14.3 GSM8K gain at {n1}: metamath_unique {gain[('metamath_unique', n1)]*100:+.1f} pt (>= +5), wild_cluster {gain[('wild_cluster', n1)]*100:+.1f} pt (<= +1); base accuracy {base:.3f}")
gu = gain[("metamath_unique", n0)]
for s in ["metamath_pad50", "metamath_followup"]:
    # [PAPER] App. P (Step 14): P14.4 Fail (105% and 118%)
    print(f"P14.4 at {n0}: {s} gain {gain[(s, n0)]*100:+.1f} pt vs unique {gu*100:+.1f} pt -> ratio {gain[(s, n0)]/gu if gu > 0 else float('nan'):.2f} (pass if <= 0.6); two-scale detection {fmt(DET[s]['twoscale'])} vs {fmt(DET['metamath_unique']['twoscale'])}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for s, c_ in [("metamath_unique", "tab:red"), ("metamath_pad50", "tab:purple"), ("metamath_followup", "tab:orange")]:
    best = max(DET[s][d] or 0 for d in ["twoscale", "mmd", "glob"])
    ax[0].scatter(gain[(s, n0)] * 100, DET[s]["twoscale"], color=c_, s=80, label=f"{s}: two-scale"); ax[0].scatter(gain[(s, n0)] * 100, best, color=c_, marker="x", s=60)
ax[0].set_xlabel(f"GSM8K gain over base (points), {n0} queries"); ax[0].set_ylabel("paired detection power (o two-scale, x best of two-scale/MMD/global)")
ax[0].set_title("Price of stealth: capability extracted vs detection"); ax[0].legend(fontsize=8)
labels = ["benign all turns", "benign first turns", "colluder unique", "colluder follow-up", "colluder natural", "colluder random"]
vals = [M1["benign_all"], M1["benign_first"], M1["colluder_metamath_unique"], M1["colluder_metamath_followup"], M1["colluder_wild_cluster"], M1["colluder_random"]]
ax[1].bar(range(len(vals)), vals, color=["tab:gray", "tab:gray", "tab:red", "tab:orange", "tab:green", "tab:blue"]); ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(labels, rotation=25, fontsize=8)
ax[1].set_ylabel("median log S_fine - log S_coarse per account"); ax[1].set_title("Fine-scale clumping (mechanism)")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step14_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 14 VERDICT REPORT"); print("=" * 90)
print("M1:", json.dumps({k: round(v, 4) for k, v in M1.items()})); print("M2:", json.dumps({k: round(v, 4) for k, v in M2.items()}))
print(f"[M3] FPR " + " ".join(f"{d}={RES_M['FPR'][d][0]:.3f}" for d in DETS))
for r in RES_M["rows"]: print(f"[M3] {r['source']:>18} K={r['K']} N~{r['N_mean']:.0f} prec={fmt(r['ts_precision'])} | " + " ".join(f"{d}={fmt(r.get(d))}[{fmt(r.get(d+'_ci', (None, None))[0])},{fmt(r.get(d+'_ci', (None, None))[1])}]" for d in DETS))
for (s, n), a in ACC.items(): print(f"[D] {s:>18} budget {n:5d}: GSM8K {a:.3f} (gain {100*(a-base):+.1f} pt)")
print("=" * 90)
with open(f"{OUTDIR}/step14_results.json", "w") as fh: json.dump(dict(M1=M1, M2=M2, M3=RES_M, ACC={f"{s}|{n}": a for (s, n), a in ACC.items()}), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
