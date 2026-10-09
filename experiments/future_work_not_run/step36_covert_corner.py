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
# ## NOT RUN — future work; no numbers in the paper come from this script
# Paper: "The Limits of Query Monitoring against Distributed Model Extraction".
#
# **NOT RUN — future work; no numbers in the paper come from this script; it implements open tests listed in §11 'Limitations and open tests'.**
# Specifically, it implements several of the experiments §11 names as future work: an anytime-valid sequential test on the hollowed score,
# colluders built from whole real user-weeks, a relay simulation, a defence against calibration poisoning, and lexical/style
# representations (§11 notes that lexical and style features are untested). The predictions P36a–P36k below were written
# in this notebook but never evaluated; they do not appear in the prediction log (docs/prediction_log.md). There is no reproduction table.

# %% [markdown]
# Step 36 —The covert corner on untouched weeks: sequential tests on the hollowed score, colluders built from whole user-weeks, relays, defender baselines outside the embedding, a defence against the calibration ramp, and hierarchical intervals
# **Kaggle:** GPU (T4/P100) + Internet + `HF_TOKEN` (WildChat-4.8M access) → **Run All** (≈6–7 h without Step 35 caches, ≈4–5 h with them). `QUICK=True` first (≈45 min). If you saved Step 35's output, attach it as an input dataset: its `step35_full_*` caches (data and MiniLM embeddings) are found automatically under `/kaggle/input`.
#
# Same untouched weeks and the same frozen pipeline as Step 35 (WildChat-4.8M, history 2 Oct 2023 – 5 May 2024, holdout 6 May – 1 Sep 2024; hashed IPs, ≥ 10 queries per account-week, 64–364-account windows, 50 colluders replacing benign accounts, identical-window pairing, conformal thresholds over all prior weeks at α = 1%).
#
# * **Colluders from whole user-weeks.** `natural_users`: 50 real attacker-pool user-weeks, unchanged (a law-matching control: if hollowed selection ranks these above benign windows, the injection itself is detectable). `tilted_users`: the 50 most on-task pool user-weeks. `written_users`: those accounts with their on-task queries replaced by LLM rewrites in the users' voice.
# * **Relays.** `relay_mix`: each colluder account replays whole pool conversations from many users (session interleaving). `relay_tilt`: the same, restricted to conversations that contain an on-task request (selection tilt).
# * **Old constructions** (Steps 27–35, accounts assembled from draws across the pool): `dataset`, `padded`, `tilted_c05`, `written_c05`; and a GSM8K evaluation sweep.
# * **Defender baselines outside MiniLM.** The frozen pipeline run on (i) a lexical representation (TF-IDF unigrams and bigrams, SVD to 256) and (ii) a style representation (length, casing, digits, punctuation, newlines, type–token ratio, GPT-2 log-perplexity); (iii) a per-account topicality test (two-sided Higher Criticism); (iv) the defender's own written-prompt discriminator, trained on rewrites it generated, as a detector; (v) MinHash near-duplicate detection across accounts.
# * **Sequential tests** (conformal test martingale and first weekly alarm) on two-scale, hollowed and multi-outlier scores, topicality and the discriminator, for every campaign.
# * **A ramp defence.** Guard-band calibration: thresholds and reference taken only from weeks more than G = 8 weeks before the test week; tested against an 8-week ramp and a 16-week ramp.
# * **Hierarchical intervals** (bootstrap over weeks, then campaign seeds within weeks) and detection stratified by window size.
#
# | ID | Prediction (written in this notebook only; never run, not evaluated, not in the paper's Prediction log) |
# |---|---|
# | **P36a** | Injection control: `natural_users` ranks at hollowed AUC ≤ 0.65 and weekly detection ≤ 0.05 on every calibrated detector |
# | **P36b** | `tilted_users` evades: every calibrated detector ≤ 0.05 weekly, and the hollowed martingale alarms within 16 weeks in ≤ 10% of runs |
# | **P36c** | The old `tilted_c05` construction is flagged by the hollowed martingale within 16 weeks in ≥ 50% of runs |
# | **P36d** | The hollowed martingale flags padded campaigns in a median ≤ 6 weeks |
# | **P36e** | Written campaigns evade hollowed selection (≤ 0.10 weekly), but the defender's discriminator detects `written_c05` ≥ 0.5 weekly at ≤ 2% false alarms |
# | **P36f** | `relay_mix` evades the embedding detectors (≤ 0.05) but per-account topicality detects it ≥ 0.3; `relay_tilt` is detected by hollowed selection ≥ 0.3 |
# | **P36g** | Lexical representation: hollowed selection detects dataset campaigns ≥ 0.8 and padded ≥ 0.3; style representation: dataset ≥ 0.5 |
# | **P36h** | MinHash near-duplicates: false alarms ≤ 2%, dataset campaigns and GSM8K sweeps ≤ 0.1 |
# | **P36i** | Guard band G = 8 restores hollowed detection against an 8-week ramp to ≥ 0.8 of its abrupt rate at ≤ 2% false alarms; a 16-week ramp cuts it again by ≥ 0.3 |
# | **P36j** | Hierarchical intervals are wider than week-only intervals; hollowed detection of padded campaigns rises with colluder share (top tercile ≥ bottom + 0.2) |
# | **P36k** | Per-account topicality and the discriminator detector are calibrated (≤ 2%) under conformal thresholds |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc, re, pickle, glob, zlib
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
USE_GPU = HAS_TORCH and DEV is not None and DEV.type == "cuda"
if not USE_GPU and not SYNTHETIC_TEST:
    raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
CHUNK = 32768 if USE_GPU else 8192
# ---- FROZEN pipeline (identical to Steps 23-35) ----
H_COARSE, H_FINE, M_MODES, ALPHA, FLOOR = 4.0, 0.5, 300, 0.01, 1e-4
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, SUB_FRAC = 500, 10, 0.2, 0.7
N_TOPICS = 100
EDGE_DELTA, MIN_PART, MAX_OUT = 0.05, 5, 20
LAGB, CAL_N, K_COLL = 4, 8, 50
SIZE_RANGE, N_FIX = (64, 364), 1640
HIST_START, HOLD_START, HOLD_END = "2023-10-02", "2024-05-06", "2024-09-02"
if QUICK: HIST_START, HOLD_END = "2024-01-29", "2024-06-03"
if QUICK:
    R = dict(NULL=10, REPS=2, SEEDS=[0, 1], SEQ_H=6, T_WEEKS=2, T_CAL=3, T_REPS=3, N_GEN=96, GEN_BS=16, MAX_NEW_GEN=80, CONV_KEEP=0.5, GUARD=8, POOL_SPAN=4)
else:
    R = dict(NULL=30, REPS=3, SEEDS=[0, 1, 2], SEQ_H=16, T_WEEKS=4, T_CAL=10, T_REPS=8, N_GEN=3000, GEN_BS=48, MAX_NEW_GEN=110, CONV_KEEP=1.0, GUARD=8, POOL_SPAN=4)
if SYNTHETIC_TEST: M_MODES = 60; SIZE_RANGE, N_FIX, K_COLL = (40, 90), 300, 10; R["GUARD"] = 4
GEN_MODEL, PPL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct", "openai-community/gpt2"
TAG = f"{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}"
CACHE = f"{OUTDIR}/step35_{TAG}"                     # Step 35 caches (data, MiniLM), reused when present
C36 = f"{OUTDIR}/step36_{TAG}"
def find(path):
    if os.path.exists(path): return path
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(path)}", recursive=True); return hits[0] if hits else None
def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")
def st_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256; return enc
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK}")

# %% [markdown]
# # Part 0 — data, texts and MiniLM embeddings

# %%
# ===== Cell 2: English user turns from HIST_START to HOLD_END (Step 35 cache if present) =====
t0 = time.time()
D0, D1, DH = np.datetime64(HIST_START), np.datetime64(HOLD_END), np.datetime64(HOLD_START)
HOLD_WEEK = int((DH - D0).astype(int) // 7)
fm, ft = find(CACHE + "_meta.npz"), find(CACHE + "_txt.pkl")
if SYNTHETIC_TEST:
    rng = np.random.default_rng(0); n_weeks, n_ip = 26, 2500; HOLD_WEEK = 18; rows_a, rows_w = [], []
    for w in range(n_weeks):
        act = rng.choice(n_ip, 500, replace=False); vol = np.minimum(3 + rng.geometric(0.08, 500), 200)
        rows_a.append(np.repeat(act, vol)); rows_w.append(np.full(vol.sum(), w))
    ACCT = np.concatenate(rows_a).astype(np.int32); WEEK = np.concatenate(rows_w).astype(np.int16)
    CONV = (np.arange(len(ACCT)) // 3).astype(np.int32); TURN = (np.arange(len(ACCT)) % 3 + 1).astype(np.int16)
    VOC = [[f"w{t}_{j}" for j in range(30)] for t in range(41)]; topic = rng.integers(0, 40, n_ip)[ACCT]
    TXT = [" ".join(rng.choice(VOC[t], int(rng.integers(4, 15)))) + ("?" if rng.random() < .5 else ".") for t in topic]
elif fm and ft:
    _c = np.load(fm); ACCT, WEEK, CONV, TURN = _c["ACCT"], _c["WEEK"], _c["CONV"], _c["TURN"]
    with open(ft, "rb") as fh: TXT = pickle.load(fh)
    print(f"loaded Step 35 cache: {len(ACCT)} turns")
else:
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    from huggingface_hub import HfFileSystem, hf_hub_download
    tok = hf_token(); fs = HfFileSystem(token=tok); REPO = "allenai/WildChat-4.8M"
    files = sorted(fs.glob(f"datasets/{REPO}/data/*.parquet")); print(f"{len(files)} parquet shards")
    def to_days(col):
        col = col.combine_chunks() if hasattr(col, "combine_chunks") else col
        if pa.types.is_timestamp(col.type):
            v = pc.cast(col, pa.int64()).fill_null(0).to_numpy(zero_copy_only=False); return v.astype(f"datetime64[{col.type.unit}]").astype("datetime64[D]")
        return np.array([str(x)[:10] if x is not None else "1970-01-01" for x in col.to_pylist()], dtype="datetime64[D]")
    TXT, IPS, DAYS, CONV_L, TURN_L = [], [], [], [], []; cid = 0; tmp = os.path.join(OUTDIR, "_wc48"); os.makedirs(tmp, exist_ok=True)
    rkeep = np.random.default_rng(SEED)
    for fi, f in enumerate(files):
        try:
            with fs.open(f, "rb") as fh: d_ = to_days(pq.ParquetFile(fh).read(columns=["timestamp"]).column(0))
            if d_.max() < D0 or d_.min() >= D1: continue
        except Exception as ex:
            print(f"  [{fi}] timestamp projection failed ({type(ex).__name__}); downloading the shard")
        local = hf_hub_download(REPO, f.split(f"{REPO}/")[1], repo_type="dataset", token=tok, local_dir=tmp)
        tbl = pq.read_table(local, columns=["conversation", "hashed_ip", "timestamp", "language"]); os.remove(local)
        d_ = to_days(tbl["timestamp"]); lang = tbl["language"].to_numpy(zero_copy_only=False); ipv = tbl["hashed_ip"].to_numpy(zero_copy_only=False)
        m = (d_ >= D0) & (d_ < D1) & (lang == "English") & np.array([x is not None for x in ipv])
        if R["CONV_KEEP"] < 1: m &= rkeep.random(len(m)) < R["CONV_KEEP"]
        if not m.any(): continue
        tbl = tbl.filter(pa.array(m)); d_, ipv = d_[m], ipv[m]
        conv = tbl["conversation"].combine_chunks(); par = pc.list_parent_indices(conv).to_numpy(); flat = pc.list_flatten(conv)
        role, content = flat.field("role"), flat.field("content")
        um = pc.fill_null(pc.and_(pc.equal(role, "user"), pc.greater(pc.utf8_length(content), 0)), False).to_numpy(zero_copy_only=False)
        pu = par[um]; txt = [s[:1000] for s in pc.filter(content, pa.array(um)).to_pylist()]
        st = np.r_[0, np.flatnonzero(np.diff(pu)) + 1]; first = np.repeat(st, np.diff(np.r_[st, len(pu)]))
        TXT += txt; IPS.append(ipv[pu]); DAYS.append(d_[pu]); CONV_L.append(cid + pu); TURN_L.append(np.arange(len(pu)) - first + 1); cid += len(tbl)
        print(f"  [{fi}] {f.split('/')[-1]}: {int(m.sum())} conversations in range, {len(pu)} user turns (total {len(TXT)}; {time.time()-t0:.0f}s)")
    _, ACCT = np.unique(np.concatenate(IPS), return_inverse=True); ACCT = ACCT.astype(np.int32)
    WEEK = ((np.concatenate(DAYS) - D0).astype(np.int64) // 7).astype(np.int16)
    CONV, TURN = np.concatenate(CONV_L).astype(np.int32), np.minimum(np.concatenate(TURN_L), 32000).astype(np.int16); del IPS, DAYS; gc.collect()
    np.savez(CACHE + "_meta.npz", ACCT=ACCT, WEEK=WEEK, CONV=CONV, TURN=TURN)
    with open(CACHE + "_txt.pkl", "wb") as fh: pickle.dump(TXT, fh)
NW_ = len(ACCT); n_w = int(WEEK.max()) + 1; n_acct = int(ACCT.max()) + 1; HIST = WEEK < HOLD_WEEK
is_pool = np.random.default_rng(SEED + 1).random(n_acct) < POOL_FRAC                      # frozen attacker-pool split
POOLROW = is_pool[ACCT]
print(f"{NW_} English user turns, {n_acct} hashed IPs, {n_w} weeks (history {HOLD_WEEK}, holdout {n_w - HOLD_WEEK}) ({time.time()-t0:.0f}s)")

# %%
# ===== Cell 3: MiniLM embeddings, external texts, task regions =====
t0 = time.time()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
if SYNTHETIC_TEST:
    r = np.random.default_rng(1); base = r.standard_normal((41, 384)); topic = r.integers(0, 40, n_acct)[ACCT]
    E = base[topic] * 0.3 + r.standard_normal((NW_, 384)) * 0.9; E /= np.linalg.norm(E, axis=1, keepdims=True); E_WC = E.astype(np.float16)
    MMQ = [f"what is {a} plus {a + 3} times two" for a in range(1500)]; MMA = [f"The answer is: {2 * a + 6}" for a in range(1500)]
    EVQ = [f"tom has {a} apples how many" for a in range(300)]; GTR = [f"sara buys {a} pens" for a in range(800)]
    GSMROWS = list(zip(MMQ, MMA))
    def embed(T):
        h = np.array([zlib.crc32(t.encode()) for t in T]); rr = np.random.default_rng(h % 1000)
        out = base[40] * 0.8 + np.stack([np.random.default_rng(int(x)).standard_normal(384) for x in h]) * 0.6; return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float16)
else:
    from datasets import load_dataset
    tok = hf_token(); enc = st_encoder()
    def embed(T): return enc.encode(T, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    fe = find(CACHE + "_minilm.npy")
    if fe: E_WC = np.load(fe); print(f"loaded Step 35 MiniLM embeddings {E_WC.shape}")
    else:
        E_WC = np.empty((NW_, 384), np.float16)
        for s0 in range(0, NW_, 400000): E_WC[s0:s0 + 400000] = embed(TXT[s0:s0 + 400000]); print(f"  embedded {min(s0 + 400000, NW_)}/{NW_} ({time.time()-t0:.0f}s)")
        np.save(CACHE + "_minilm.npy", E_WC)
    mmd = load_dataset("meta-math/MetaMathQA", split="train", token=tok).select_columns(["query", "response", "original_question", "type"]); groups = {}
    GSMROWS = []
    for q, a, o, ty in zip(mmd["query"], mmd["response"], mmd["original_question"], mmd["type"]):
        if q and a and o:
            groups.setdefault(o, []).append((q.strip()[:1000], a.strip()[:2000]))
            if ty and ty.startswith("GSM"): GSMROWS.append((q.strip()[:1000], a.strip()[:2000]))
    rr = np.random.default_rng(SEED); origs = list(groups); rr.shuffle(origs)
    MMP = [groups[o][int(rr.integers(len(groups[o])))] for o in origs]; MMQ = [q for q, _ in MMP]; MMA = [a for _, a in MMP]
    g_te = load_dataset("openai/gsm8k", "main", split="test"); g_tr = load_dataset("openai/gsm8k", "main", split="train")
    EVQ = [q[:1000] for q in g_te["question"]]; GTR = [q[:1000] for q in g_tr["question"]]
E_MM, E_EV, E_GTR = embed(MMQ), embed(EVQ), embed(GTR)
ben_h8 = np.flatnonzero(~POOLROW & (WEEK < 8))
def region_clf(Epos, seed):
    neg = np.random.default_rng(seed).choice(ben_h8, min(4 * len(Epos), len(ben_h8)), replace=False)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([Epos.astype(np.float32), E_WC[neg].astype(np.float32)]), np.r_[np.ones(len(Epos)), np.zeros(len(neg))])
def post_of(clf, E_): return np.concatenate([clf.predict_proba(E_[s0:s0 + 200000].astype(np.float32))[:, 1] for s0 in range(0, len(E_), 200000)])
clf_s, clf_v = region_clf(E_MM, 4), region_clf(E_GTR, 5)
PS_, PV_ = post_of(clf_s, E_WC), post_of(clf_v, E_WC)
IN_S, IN_V = PS_ >= 0.5, PV_ >= 0.5                                                          # MetaMathQA-like region (tilt); GSM8K value region (writer)
for nm_, P_, IN_ in [("S", PS_, IN_S), ("V", PV_, IN_V)]:
    if (IN_ & POOLROW).sum() < 200 or (IN_ & ~POOLROW & HIST).sum() < 200:
        IN_[:] = P_ >= np.quantile(P_, 0.98); print(f"[region {nm_}] WARNING: few on-task turns; using the top 2% by posterior")
P_TASK = float(IN_S[~POOLROW & HIST].mean())
print(f"task share p = {P_TASK:.4f}; value-region share {IN_V[~POOLROW & HIST].mean():.4f}; on-task pool turns {int((IN_S & POOLROW).sum())} ({time.time()-t0:.0f}s)")

# %%
# ===== Cell 4: written prompts in the users' voice (attacker half and defender half), cached =====
t0 = time.time(); fw = find(C36 + "_written.pkl")
if fw:
    with open(fw, "rb") as fh: RW_TXT, RW_ANS, ATT_IX, DEF_IX, RW_PASS, RW_AUC = pickle.load(fh)
else:
    style_rows = np.flatnonzero(IN_V & POOLROW & HIST); style = [TXT[i][:220].replace("\n", " ") for i in style_rows] or ["can you help me with this math question"]
    src = [GSMROWS[i] for i in np.random.default_rng(32).choice(len(GSMROWS), min(R["N_GEN"], len(GSMROWS)), replace=False)]
    if SYNTHETIC_TEST:
        def regen(qs, strong): return ["hey can u help me " + q for q in qs]
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        gtok = AutoTokenizer.from_pretrained(GEN_MODEL); gtok.padding_side = "left"
        gm = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float16).to(DEV); gm.eval(); rr = np.random.default_rng(7)
        def regen(qs, strong):
            outs = []
            for s0 in range(0, len(qs), R["GEN_BS"]):
                msgs = []
                for q in qs[s0:s0 + R["GEN_BS"]]:
                    ex = "\n".join(f"- {style[i]}" for i in rr.choice(len(style), 3, replace=len(style) < 3))
                    extra = " Match their length, casing and punctuation closely; do not sound like a textbook or a worksheet." if strong else ""
                    msgs.append(gtok.apply_chat_template([{"role": "user", "content": "Here are messages that real people typed into a chatbot when asking for help with math:\n" + ex +
                        "\n\nRewrite the following problem as a message from one of these people, in their style and voice (informal, maybe a little context about why they ask). Keep every number and exactly what is being asked." + extra + " Output only the message.\n\nProblem: " + q}], tokenize=False, add_generation_prompt=True))
                ids = gtok(msgs, return_tensors="pt", padding=True).to(DEV)
                with torch.no_grad(): out = gm.generate(**ids, max_new_tokens=R["MAX_NEW_GEN"], do_sample=True, temperature=0.9 if strong else 0.8, top_p=0.95, pad_token_id=gtok.eos_token_id)
                outs += [gtok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n\n")[0][:1000] for o in out]
                if s0 % (10 * R["GEN_BS"]) == 0: print(f"  [W] {len(outs)}/{len(qs)} e.g. {outs[-1][:100]!r}")
            return outs
    RW_TXT = regen([q for q, _ in src], False); RW_ANS = [a for _, a in src]
    perm = np.random.default_rng(33).permutation(len(RW_TXT)); ATT_IX, DEF_IX = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    real_on = np.flatnonzero(IN_V & POOLROW); real = E_WC[np.random.default_rng(8).choice(real_on, min(len(real_on), len(ATT_IX)), replace=len(real_on) < len(ATT_IX))].astype(np.float32)
    def crossfit(Dw):
        iw, ir = np.random.default_rng(9).permutation(len(Dw)), np.random.default_rng(10).permutation(len(real)); P = np.zeros(len(Dw)); au = []
        for f in range(2):
            tw, hw, tr, hr = iw[f::2], iw[1 - f::2], ir[f::2], ir[1 - f::2]
            dc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([Dw[tw], real[tr]]), np.r_[np.ones(len(tw)), np.zeros(len(tr))])
            P[hw] = dc.predict_proba(Dw[hw])[:, 1]; au.append(roc_auc_score(np.r_[np.ones(len(hw)), np.zeros(len(hr))], np.r_[P[hw], dc.predict_proba(real[hr])[:, 1]]))
        return P, float(np.mean(au))
    DA = embed([RW_TXT[i] for i in ATT_IX]).astype(np.float32); P_att, auc0 = crossfit(DA); RW_AUC = [auc0]
    bad = np.flatnonzero(P_att >= 0.5)                                                     # one regeneration round for what the attacker's discriminator catches
    if len(bad):
        new = regen([src[ATT_IX[i]][0] for i in bad], True)
        for j, i in enumerate(bad): RW_TXT[ATT_IX[i]] = new[j]
        DA[bad] = embed(new).astype(np.float32); P_att, auc1 = crossfit(DA); RW_AUC.append(auc1)
    keep = (P_att < 0.5) & (clf_v.predict_proba(DA)[:, 1] >= 0.5)
    if keep.sum() < 100: keep = P_att <= np.quantile(P_att, 0.5); print("  [W] FILTER RELAXED: using the least-distinguishable half")
    RW_PASS = ATT_IX[keep]
    if not SYNTHETIC_TEST: del gm; gc.collect(); torch.cuda.empty_cache()
    with open(C36 + "_written.pkl", "wb") as fh: pickle.dump((RW_TXT, RW_ANS, ATT_IX, DEF_IX, RW_PASS, RW_AUC), fh)
print(f"[W] rewrites {len(RW_TXT)} (attacker {len(ATT_IX)}, defender {len(DEF_IX)}); attacker discriminator AUC by round {[round(x, 3) for x in RW_AUC]}; passing {len(RW_PASS)} ({time.time()-t0:.0f}s)")
# ---- the item space: WildChat turns, then MetaMathQA, GSM8K test, rewrites
OFF = {"mm": NW_, "ev": NW_ + len(MMQ), "rw": NW_ + len(MMQ) + len(EVQ)}; N_ITEMS = OFF["rw"] + len(RW_TXT)
TXT_ALL = TXT + MMQ + EVQ + RW_TXT; E_RW = embed(RW_TXT)
EMB = {"minilm": np.vstack([E_WC, E_MM, E_EV, E_RW])}
ITEM_MM = OFF["mm"] + np.arange(len(MMQ)); ITEM_EV = OFF["ev"] + np.arange(len(EVQ)); ITEM_RWP = OFF["rw"] + RW_PASS
# defender's discriminator: its own rewrites (defender half) against real on-task prompts of benign history accounts
real_def = np.flatnonzero(IN_V & ~POOLROW & HIST)
real_def = np.random.default_rng(12).choice(real_def, min(len(real_def), 4 * len(DEF_IX)), replace=len(real_def) < 4 * len(DEF_IX))
clf_d = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([E_RW[DEF_IX].astype(np.float32), E_WC[real_def].astype(np.float32)]), np.r_[np.ones(len(DEF_IX)), np.zeros(len(real_def))])
PD = post_of(clf_d, EMB["minilm"]).astype(np.float32)
with open(C36 + "_prompts.pkl", "wb") as fh:                                              # for Step 37 (distillation with a live teacher)
    on_rows = np.flatnonzero(IN_S & POOLROW); off_rows = np.flatnonzero(~IN_S & POOLROW)
    pickle.dump(dict(mm=MMP if not SYNTHETIC_TEST else GSMROWS, gsm_rows=GSMROWS[:20000], on_task=[TXT[i] for i in on_rows], off_task=[TXT[i] for i in np.random.default_rng(1).choice(off_rows, min(30000, len(off_rows)), replace=False)],
                     written=[(RW_TXT[i], RW_ANS[i]) for i in RW_PASS], written_all_att=[(RW_TXT[i], RW_ANS[i]) for i in ATT_IX]), fh)
print(f"[D] defender discriminator: mean posterior on passing rewrites {PD[ITEM_RWP].mean():.3f}, on benign history turns {PD[:NW_][~POOLROW & HIST].mean():.3f}")

# %%
# ===== Cell 5: representations outside MiniLM — lexical (TF-IDF + SVD), style (+ GPT-2 log-perplexity); MinHash signatures =====
t0 = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
ben_hist = np.flatnonzero(~POOLROW & HIST)
fl = find(C36 + "_lexical.npy")
if fl: EMB["lexical"] = np.load(fl)
else:
    fit_ix = np.random.default_rng(21).choice(ben_hist, min(150000, len(ben_hist)), replace=False)
    vec = TfidfVectorizer(max_features=100000, min_df=2 if SYNTHETIC_TEST else 5, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32).fit([TXT_ALL[i] for i in fit_ix])
    ncomp = 32 if SYNTHETIC_TEST else 256
    svd = TruncatedSVD(ncomp, random_state=SEED).fit(vec.transform([TXT_ALL[i] for i in fit_ix[:100000]]))
    L = np.empty((N_ITEMS, ncomp), np.float16)
    for s0 in range(0, N_ITEMS, 100000):
        x = svd.transform(vec.transform(TXT_ALL[s0:s0 + 100000])); L[s0:s0 + 100000] = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)).astype(np.float16)
    EMB["lexical"] = L; np.save(C36 + "_lexical.npy", L)
print(f"[lexical] {EMB['lexical'].shape} ({time.time()-t0:.0f}s)")
fs_ = find(C36 + "_style.npy")
if fs_: EMB["style"] = np.load(fs_)
else:
    def feats(t):
        L_ = len(t); w = t.split(); nw = max(len(w), 1); letters = sum(c.isalpha() for c in t) or 1
        return [math.log1p(L_), math.log1p(len(w)), sum(c.isupper() for c in t) / letters, sum(c.isdigit() for c in t) / max(L_, 1),
                sum(c in ".,;:!?()[]{}\"'-" for c in t) / max(L_, 1), math.log1p(t.count("\n")), float("?" in t), float(t[:1].isupper()),
                float(t.rstrip()[-1:] in ".?!"), len(set(w)) / nw, sum(len(x) for x in w) / nw, sum(ord(c) > 127 for c in t) / max(L_, 1)]
    F = np.array([feats(t) for t in TXT_ALL], np.float32)
    if SYNTHETIC_TEST: nll = np.random.default_rng(3).standard_normal(N_ITEMS).astype(np.float32)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        ptok = AutoTokenizer.from_pretrained(PPL_MODEL); ptok.pad_token = ptok.eos_token; pm = AutoModelForCausalLM.from_pretrained(PPL_MODEL, dtype=torch.float16).to(DEV).eval()
        nll = np.zeros(N_ITEMS, np.float32)
        for s0 in range(0, N_ITEMS, 512):
            b = ptok(TXT_ALL[s0:s0 + 512], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
            with torch.no_grad(): lg = pm(**b).logits[:, :-1].float()
            tgt = b["input_ids"][:, 1:]; msk = b["attention_mask"][:, 1:].float()
            lp = torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
            nll[s0:s0 + 512] = ((lp * msk).sum(1) / msk.sum(1).clamp(min=1)).cpu().numpy()
            if s0 % 200000 == 0: print(f"  [ppl] {s0}/{N_ITEMS} ({time.time()-t0:.0f}s)")
        del pm; torch.cuda.empty_cache()
    F = np.c_[F, nll]; mu, sd = F[ben_hist].mean(0), F[ben_hist].std(0) + 1e-6; EMB["style"] = ((F - mu) / sd).astype(np.float16); np.save(C36 + "_style.npy", EMB["style"])
print(f"[style] {EMB['style'].shape} ({time.time()-t0:.0f}s)")
fh_ = find(C36 + "_minhash.npy")
NB, RB = 8, 8; PR = np.uint64(4294967311); rs = np.random.default_rng(41)
MA_, MB_ = rs.integers(1, 2 ** 32 - 1, NB * RB).astype(np.uint64), rs.integers(0, 2 ** 32 - 1, NB * RB).astype(np.uint64)
if fh_: BK = np.load(fh_)
else:
    BK = np.empty((N_ITEMS, NB), np.uint64)
    for i, t in enumerate(TXT_ALL):
        w = re.findall(r"\w+", t.lower()); sh = [" ".join(w[j:j + 3]) for j in range(max(1, len(w) - 2))] or [t]
        x = np.array([zlib.crc32(s_.encode()) for s_ in sh], np.uint64)
        sig = ((np.outer(x, MA_) + MB_) % PR).min(0).reshape(NB, RB)
        BK[i] = np.array([zlib.crc32(row.tobytes()) for row in sig], np.uint64)
    np.save(C36 + "_minhash.npy", BK)
print(f"[minhash] band keys {BK.shape} ({time.time()-t0:.0f}s)")
REPS_ = ["minilm", "lexical", "style"]

# %% [markdown]
# # Part 1 — frozen statistics, extra detectors, campaigns

# %%
# ===== Cell 6: frozen statistics (verbatim from Steps 23-35) =====
from functools import lru_cache
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
@lru_cache(maxsize=64)
def mp_median(c):
    a, b = (1 - np.sqrt(c)) ** 2, (1 + np.sqrt(c)) ** 2; x = np.linspace(a + 1e-9, b - 1e-9, 20001)
    f = np.sqrt(np.maximum((b - x) * (x - a), 0)) / (2 * np.pi * c * x); F = np.cumsum(f) * (x[1] - x[0]); F /= F[-1]
    return float(x[np.searchsorted(F, 0.5)])
def top_group(v):
    v2 = v ** 2 / (v ** 2).sum(); o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); return o[:m]
def ts_of(Yc, Yf, cnt, g): return float(np.log(union_S(Yc, cnt, g)) - np.log(union_S(Yf, cnt, g)))
def gram(Yn, kk=50):
    A_, r_ = Yn.shape
    if A_ >= r_:
        ev, U = np.linalg.eigh(Yn.T @ Yn); o = np.argsort(-ev); ev, U = ev[o], U[:, o]
        V = Yn @ U[:, :kk] / np.sqrt(np.maximum(ev[:kk], 1e-12)); return ev, V
    ev, V = np.linalg.eigh(Yn @ Yn.T); o = np.argsort(-ev); return ev[o], V[:, o][:, :kk]
def stats_rep(pts, labs, cnt, Bc, Bf):
    Yc = whitened(pts, labs, cnt, Bc); Yf = whitened(pts, labs, cnt, Bf); A_, r_ = Yc.shape
    ev, Vc = np.linalg.eigh(Yc.T @ Yc / r_); gC = top_group(Yc @ Vc[:, -1])
    Ycc = Yc - Yc.mean(0, keepdims=True); Yn = Ycc / np.maximum(np.linalg.norm(Ycc, axis=1, keepdims=True), 1e-12)
    lam, V = gram(Yn); nz = lam[: min(A_, r_)]; g_ = A_ / r_
    e_hat = float(np.median(nz) / (max(g_, 1.0) * mp_median(min(g_, 1 / g_))) * (1 + np.sqrt(g_)) ** 2 - 1)
    J = [0]
    for j in range(min(MAX_OUT, V.shape[1])):
        if lam[j] - 1 > (1 + EDGE_DELTA) * e_hat and j > 0 and 1.0 / np.sum(V[:, j] ** 4) >= MIN_PART: J.append(j)
    groups = [top_group(V[:, j]) for j in J]; sc = [ts_of(Yc, Yf, cnt, g) for g in groups]; jm = int(np.argmax(sc))
    return dict(tsC=ts_of(Yc, Yf, cnt, gC), tsRc=sc[0], Tmax=float(max(sc)), gC=gC, gR=groups[0], gT=groups[jm], accS=np.log(np.maximum((Yc ** 2).mean(1), 1e-9)))
def boot_ci(x, B=1000, rng=np.random.default_rng(7)):
    x = np.array(x, float)
    if len(x) == 0: return (float("nan"),) * 3
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
def hboot(nested, B=1000, rng=np.random.default_rng(8)):
    """nested: {week: {seed: [values]}} -> mean and 95% interval resampling weeks, then seeds within weeks, then windows."""
    ws = [w for w in nested if nested[w]]; out = []
    for _ in range(B):
        vals = []
        for w in rng.choice(ws, len(ws)):
            ss = list(nested[w]);
            for s_ in rng.choice(ss, len(ss)):
                v = np.array(nested[w][s_]); vals.append(v[rng.integers(0, len(v), len(v))].mean())
        out.append(np.mean(vals))
    flat = [np.mean(nested[w][s_]) for w in ws for s_ in nested[w]]
    return float(np.mean(flat)), float(np.quantile(out, .025)), float(np.quantile(out, .975))
def roc(null_vals, att_vals):
    null_vals, att_vals = np.array(null_vals), np.array(att_vals)
    auc = float(np.mean(att_vals[:, None] > null_vals[None, :]) + 0.5 * np.mean(att_vals[:, None] == null_vals[None, :]))
    return dict(auc=auc, tpr1=float(np.mean(att_vals > np.quantile(null_vals, 0.99))))
print("frozen statistics defined")

# %%
# ===== Cell 7: projections, weekly groups, lagged banks for every representation =====
t0 = time.time(); Z = {}
for rep in REPS_:
    X = EMB[rep]; D_ = X.shape[1]; fit = X[np.random.default_rng(SEED).choice(ben_hist, min(100000, len(ben_hist)), replace=False)].astype(np.float32)
    MU = fit.mean(0); ew, U = np.linalg.eigh(np.cov(fit - MU, rowvar=False)); EW, UB = ew[::-1], U[:, ::-1].copy()
    W_ = UB / math.sqrt(max(EW.sum(), 1e-9) / 4.0); Zr = np.empty((N_ITEMS, D_), np.float32)
    for s0 in range(0, N_ITEMS, 200000): Zr[s0:s0 + 200000] = (X[s0:s0 + 200000].astype(np.float32) - MU) @ W_
    Z[rep] = Zr
def groups_by_week(pool_flag, min_q):
    order = np.lexsort((ACCT, WEEK)); keys = WEEK[order].astype(np.int64) * n_acct + ACCT[order]; GA = {b: [] for b in range(n_w)}
    for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
        if bool(is_pool[int(ACCT[g[0]])]) == pool_flag and len(g) >= min_q: GA[int(WEEK[g[0]])].append(g[:VOL_CAP])
    return GA
GAM, GAP = groups_by_week(False, MIN_PER_ACCT), groups_by_week(True, MIN_PER_ACCT)
pr = np.flatnonzero(POOLROW); co = np.lexsort((TURN[pr], CONV[pr])); pr = pr[co]
CONVS = [g for g in np.split(pr, np.flatnonzero(np.diff(CONV[pr])) + 1)]                 # pool conversations, turns in order
CONV_W = np.array([int(WEEK[g[0]]) for g in CONVS]); CONV_ON = np.array([bool(IN_S[g].any()) for g in CONVS])
BANK = {rep: {} for rep in REPS_}
for b in range(1, n_w):
    prev = [np.concatenate(GAM[v]) for v in range(max(0, b - LAGB), b) if GAM[v]]
    if not prev: continue
    ref = np.concatenate(prev); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
    for rep in REPS_: BANK[rep][b] = (make_bank_aug(Z[rep][ref], H_COARSE, SEED + 7), make_bank_aug(Z[rep][ref], H_FINE, SEED + 8))
print(f"projections and banks for {len(BANK['minilm'])} weeks; pool conversations {len(CONVS)} ({int(CONV_ON.sum())} with an on-task request) ({time.time()-t0:.0f}s)")
def sub_window(b, r):
    G = GAM[b]; k = min(max(20, int(SUB_FRAC * len(G))), int(r.integers(SIZE_RANGE[0], SIZE_RANGE[1] + 1)))
    pick = r.choice(len(G), k, replace=False); parts = [G[i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols


# %%
# ===== Cell 8: window evaluation across representations; topicality, discriminator and near-duplicate detectors =====
NBINS = [10, 20, 50, 10 ** 9]
def dup_accounts(items, labs):
    hit = np.zeros(int(labs.max()) + 1, bool)
    for bnd in range(NB):
        k = BK[items, bnd]; o = np.lexsort((labs, k)); ks, ls = k[o], labs[o]
        new = np.r_[True, ks[1:] != ks[:-1]]; gid = np.cumsum(new) - 1
        nl = np.bincount(gid, weights=np.r_[True, (ks[1:] != ks[:-1]) | (ls[1:] != ls[:-1])].astype(float))   # distinct accounts per key
        multi = nl[gid] >= 2; hit[ls[multi]] = True
    return float(hit.sum())
def evaluate(items, labs, b, banks=None, reps=REPS_):
    cnt = np.bincount(labs); out = dict(A=len(cnt))
    for rep in reps:
        bk = (banks or {}).get(rep, BANK[rep][b]); s = stats_rep(Z[rep][items], labs, cnt, *bk)
        for k in ("tsC", "tsRc", "Tmax"): out[f"{k}_{rep}"] = s[k]
        if rep == "minilm": out.update(gC=s["gC"], gR=s["gR"], gT=s["gT"], accS=s["accS"], accN=cnt)
    if "minilm" in reps:
        pm = np.bincount(labs, weights=PD[items]) / cnt; out["disc"] = float(pm[cnt >= MIN_PER_ACCT].max()) if (cnt >= MIN_PER_ACCT).any() else 0.0
        out["dup"] = dup_accounts(items, labs)
    return out
REFT = None
def topic_hc(s):
    ps = []
    for lo, hi in zip(NBINS[:-1], NBINS[1:]):
        m = (s["accN"] >= lo) & (s["accN"] < hi); Rf = REFT[(lo, hi)]
        if m.sum() == 0 or len(Rf) < 20: continue
        u = (np.searchsorted(Rf, s["accS"][m]) + 0.5) / (len(Rf) + 1); ps.append(np.minimum(u, 1 - u) * 2)
    if not ps: return 0.0
    p = np.sort(np.concatenate(ps)); n = len(p); i = np.arange(1, n + 1); k = max(1, n // 2)
    return float(np.max(np.sqrt(n) * (i[:k] / n - p[:k]) / np.sqrt(np.maximum(p[:k] * (1 - p[:k]), 1e-12))))
DETS = [f"{k}_{rep}" for rep in REPS_ for k in ("tsC", "tsRc", "Tmax")] + ["topic", "disc", "dup"]
KEY = ["tsC_minilm", "tsRc_minilm", "Tmax_minilm", "tsRc_lexical", "tsRc_style", "topic", "disc", "dup"]

# %%
# ===== Cell 9: null windows and conformal thresholds =====
t0 = time.time(); NUL = {}
for b in sorted(BANK["minilm"]):
    if len(GAM[b]) < 20: continue
    NUL[b] = []
    for t in range(R["NULL"]):
        i, l, v = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); NUL[b].append(evaluate(i, l, b))
    if b % 8 == 0: print(f"  nulls through week {b} ({time.time()-t0:.0f}s)")
early = [s for b in NUL if b < 8 for s in NUL[b]]                                            # fixed topicality reference: first 8 weeks
REFT = {(lo, hi): np.sort(np.concatenate([s["accS"][(s["accN"] >= lo) & (s["accN"] < hi)] for s in early])) for lo, hi in zip(NBINS[:-1], NBINS[1:])}
for b in NUL:
    for s in NUL[b]: s["topic"] = topic_hc(s)
TH = {}
for b in sorted(NUL):
    prior = [s for v in NUL if 8 <= v < b for s in NUL[v]]
    if len([v for v in NUL if 8 <= v < b]) < CAL_N: continue
    TH[b] = {d: float(np.quantile([s[d] for s in prior], 1 - ALPHA)) for d in DETS}
HOLDW = [b for b in sorted(TH) if b >= HOLD_WEEK and len(GAM[b]) >= 40]
FPR = {d: boot_ci([np.mean([s[d] > TH[b][d] for s in NUL[b]]) for b in HOLDW]) for d in DETS}
print(f"[null] holdout weeks {HOLDW} ({time.time()-t0:.0f}s)")
print("[P36k] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.4f}" for d in DETS))


# %%
# ===== Cell 10: campaigns as lists of item ids per colluder account =====
def vols_for(b, r, K_, n_fix=None):
    cv = r.choice(np.array([len(g) for g in GAM[b]]), K_, replace=True)
    if n_fix: cv = np.maximum(1, np.round(cv * n_fix / cv.sum())).astype(int)
    return cv
def split_by(items, cv): return np.split(items[: int(np.sum(cv))], np.cumsum(cv)[:-1])
def interleave(a, b_):
    out = np.empty(len(a) + len(b_), np.int64); out[0::2] = a[: (len(out) + 1) // 2]; out[1::2] = b_[: len(out) // 2]; return out
def pool_userweeks(b):
    return [g for v in range(max(0, b - R["POOL_SPAN"] + 1), b + 1) for g in GAP[v]]
def campaign(src, b, r, K_=None):
    K_ = K_ or K_COLL; pr_ = np.flatnonzero(POOLROW)
    if src in ("natural_users", "tilted_users", "written_users"):
        U = pool_userweeks(b)
        if len(U) < K_: return None
        if src == "natural_users": return [U[i] for i in r.choice(len(U), K_, replace=False)]
        sh = np.array([IN_S[g].mean() for g in U]); top = [U[i] for i in np.argsort(-sh)[:K_]]
        if src == "tilted_users": return top
        out = []
        for g in top:
            g = g.copy(); on = IN_S[g]; g[on] = r.choice(ITEM_RWP, int(on.sum()), replace=int(on.sum()) > len(ITEM_RWP)); out.append(g)
        return out
    if src in ("relay_mix", "relay_tilt"):
        cv = vols_for(b, r, K_); ok = np.flatnonzero((CONV_W > b - R["POOL_SPAN"]) & (CONV_W <= b) & (CONV_ON if src == "relay_tilt" else True))
        if len(ok) < K_: return None
        out, order_, ptr = [], r.permutation(ok), 0                                         # each relayed conversation is used once
        for a in range(K_):
            acc, n = [], 0
            while n < cv[a] and ptr < len(order_): g = CONVS[order_[ptr]]; ptr += 1; acc.append(g); n += len(g)
            if not acc: return None
            out.append(np.concatenate(acc)[: cv[a]])
        return out
    if src.startswith("eval"): return np.split(r.permutation(ITEM_EV)[:1000], 10)                  # a benchmark partitioned across 10 accounts
    cv = vols_for(b, r, K_); n = int(cv.sum()); nh = (n + 1) // 2
    on_pool, off_pool = pr_[IN_S[pr_]], pr_[~IN_S[pr_]]
    if src == "dataset": items = r.choice(ITEM_MM, n, replace=n > len(ITEM_MM))
    elif src == "padded": items = interleave(r.choice(ITEM_MM, nh, replace=False), r.choice(pr_, n - nh, replace=False))
    elif src == "tilted_c05": items = interleave(r.choice(on_pool, nh, replace=nh > len(on_pool)), r.choice(off_pool, n - nh, replace=False))
    elif src == "written_c05": items = interleave(r.choice(ITEM_RWP, nh, replace=nh > len(ITEM_RWP)), r.choice(off_pool, n - nh, replace=False))
    elif src == "random_draws": items = r.choice(pr_, n, replace=False)
    else: raise ValueError(src)
    return split_by(items, cv)
def attacked(b, src, r, rc, banks=None, reps=REPS_, camp=None):
    idx, lab, _ = sub_window(b, r); camp = camp if camp is not None else campaign(src, b, rc)
    if camp is None: return None
    K_ = len(camp); A_ = int(lab.max()) + 1
    if K_ >= A_ // 2: return None
    keep = lab >= K_; ci = np.concatenate(camp); cl = np.concatenate([np.full(len(c), a) for a, c in enumerate(camp)])
    _, inv = np.unique(np.concatenate([cl, lab[keep]]), return_inverse=True); items = np.concatenate([ci, idx[keep]])
    sa = evaluate(items, inv, b, banks, reps); sb = evaluate(idx, lab, b, banks, reps)
    if "minilm" in reps and REFT is not None: sa["topic"], sb["topic"] = topic_hc(sa), topic_hc(sb)
    return sa, sb, K_


# %% [markdown]
# # Part 2 — weekly detection on untouched weeks

# %%
# ===== Cell 11: campaigns on holdout weeks — deployed, oracle, hierarchical intervals, strata by window size =====
t0 = time.time()
SRC_MULTI = ["dataset", "padded", "tilted_c05", "natural_users", "tilted_users"]
SRC_ONE = ["written_c05", "written_users", "relay_mix", "relay_tilt", "random_draws", "eval_gsm8k"]
RES, RAW = {}, {}
for src in SRC_MULTI + SRC_ONE:
    seeds = R["SEEDS"] if src in SRC_MULTI else R["SEEDS"][:1]; rec = []
    for sd in seeds:
        for b in HOLDW:
            for rep in range(R["REPS"]):
                r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + 7919 * sd); rc = np.random.default_rng(SEED + 17 * b + rep + 7919 * sd)
                out = attacked(b, src, r, rc)
                if out is None: continue
                sa, sb, K_ = out; row = dict(b=b, seed=sd, A=sa["A"])
                for d in DETS: row[d] = float(sa[d] > TH[b][d] and not sb[d] > TH[b][d]); row[d + "_a"] = sa[d]; row[d + "_n"] = sb[d]
                if sa["Tmax_minilm"] > TH[b]["Tmax_minilm"]: row["prec"] = float(np.mean(sa["gT"] < K_))
                rec.append(row)
    RAW[src] = rec
    res = {}
    for d in DETS:
        nested = {}
        for x in rec: nested.setdefault(x["b"], {}).setdefault(x["seed"], []).append(x[d])
        wk = boot_ci([np.mean([x[d] for x in rec if x["b"] == b]) for b in HOLDW if any(x["b"] == b for x in rec)])
        res[d] = dict(mean=wk[0], week_ci=wk[1:], h_ci=hboot(nested)[1:], **roc([x[d + "_n"] for x in rec], [x[d + "_a"] for x in rec]))
    if rec:
        A_ = np.array([x["A"] for x in rec]); q1, q2 = np.quantile(A_, [1 / 3, 2 / 3])
        res["strata"] = {d: [float(np.mean([x[d] for x in rec if lo <= x["A"] <= hi])) if any(lo <= x["A"] <= hi for x in rec) else float("nan") for lo, hi in [(q2, 1e9), (q1, q2), (0, q1)]] for d in ["tsC_minilm", "tsRc_minilm", "Tmax_minilm"]}
        res["share_terciles"] = [float(K_COLL / np.median(A_[A_ >= q2])), float(K_COLL / np.median(A_[(A_ >= q1) & (A_ < q2)])), float(K_COLL / np.median(A_[A_ < q1]))]
    RES[src] = res
    print(f"[W] {src:>14} (n={len(rec)}) | " + " ".join(f"{d}={res[d]['mean']:.2f}" for d in KEY) + " | AUC " + " ".join(f"{d}={res[d]['auc']:.2f}" for d in ["tsRc_minilm", "Tmax_minilm", "topic", "disc"]) + f"  ({time.time()-t0:.0f}s)")

# %%
# ===== Cell 12: sequential detection on every campaign (martingale and first weekly alarm), hollowed included =====
t0 = time.time(); ALPHA_SEQ = 0.01
def calib_e(p):
    p = np.clip(p, 1e-12, 1.0)
    with np.errstate(all="ignore"): return np.where(p >= 1.0, 0.0, (1 - p + p * np.log(p)) / (p * np.log(p) ** 2))
SEQW = sorted(TH); SEQD = ["tsC_minilm", "tsRc_minilm", "Tmax_minilm", "topic", "disc"]
PRIOR = {b: {d: np.array([s[d] for v in NUL if 8 <= v < b for s in NUL[v]]) for d in SEQD} for b in SEQW}
def pval(x, b, d): cal = PRIOR[b][d]; return float((1 + np.sum(cal >= x)) / (len(cal) + 1))
SEQ_SC = {"null": {b: NUL[b][0] for b in SEQW}}
for src in ["dataset", "padded", "tilted_c05", "tilted_users", "natural_users", "written_c05", "written_users", "relay_mix", "relay_tilt"]:
    SEQ_SC[src] = {}
    for b in SEQW:
        out = attacked(b, src, np.random.default_rng(SEED + 12_000 + b), np.random.default_rng(SEED + 12_500 + b), reps=["minilm"])
        if out is not None: SEQ_SC[src][b] = out[0]
def run_times(sc, start, H, d):
    M, tM, tF = 1.0, None, None
    for k, w in enumerate([w for w in SEQW if w >= start][:H]):
        if w not in sc: continue
        M *= float(calib_e(pval(sc[w][d], w, d)))
        if tM is None and M >= 1 / ALPHA_SEQ: tM = k + 1
        if tF is None and sc[w][d] > TH[w][d]: tF = k + 1
    return tM, tF
SEQ = {}
starts_all = [w for w in SEQW if len([x for x in SEQW if x >= w]) >= R["SEQ_H"]]
for src in SEQ_SC:
    for d in SEQD:
        tt = [run_times(SEQ_SC[src], s0, R["SEQ_H"], d) for s0 in starts_all]
        m_ = [t[0] if t[0] else np.inf for t in tt]; f_ = [t[1] if t[1] else np.inf for t in tt]
        th_ = run_times(SEQ_SC[src], HOLD_WEEK, len([w for w in SEQW if w >= HOLD_WEEK]), d)
        SEQ[(src, d)] = dict(m_med=float(np.median(m_)) if m_ else float("nan"), m_frac=float(np.mean(np.isfinite(m_))) if m_ else float("nan"),
                             f_med=float(np.median(f_)) if f_ else float("nan"), f_frac=float(np.mean(np.isfinite(f_))) if f_ else float("nan"), hold_m=th_[0], hold_f=th_[1], runs=len(starts_all))
    print(f"[SEQ] {src:>14}: " + " | ".join(f"{d.split('_')[0]} M {SEQ[(src, d)]['m_med']} ({SEQ[(src, d)]['m_frac']:.2f}) F {SEQ[(src, d)]['f_med']} ({SEQ[(src, d)]['f_frac']:.2f})" for d in SEQD))
print(f"[SEQ] {len(starts_all)} runs of {R['SEQ_H']} weeks ({time.time()-t0:.0f}s)")

# %% [markdown]
# # Part 3 — a defence against the calibration ramp

# %%
# ===== Cell 13: guard-band calibration against 8- and 16-week ramps =====
t0 = time.time(); G_ = R["GUARD"]; RD = ["tsC_minilm", "tsRc_minilm", "Tmax_minilm"]
def bank_from(weeks_, extra=None):
    prev = [np.concatenate(GAM[v]) for v in weeks_ if v >= 0 and GAM[v]]
    ref = np.random.default_rng(int(sum(weeks_)) + 1).choice(np.concatenate(prev), 100000 if not (QUICK or SYNTHETIC_TEST) else 5000, replace=True)
    Zr = Z["minilm"][ref] if extra is None or len(extra) == 0 else np.vstack([Z["minilm"][ref], Z["minilm"][extra]])
    return {"minilm": (make_bank_aug(Zr, H_COARSE, SEED + 7), make_bank_aug(Zr, H_FINE, SEED + 8))}
def ramp_vol(j, Lr): return (1 - j / Lr) * N_FIX if 1 <= j <= Lr else 0.0
TW = [b for b in HOLDW if b - 16 - G_ - LAGB >= 8][:: max(1, len(HOLDW) // R["T_WEEKS"])][: R["T_WEEKS"]] or HOLDW[-R["T_WEEKS"]:]
RAMP = {}
for (Lr, G) in [(8, 0), (8, G_), (16, G_)]:
    det = {d: {"abrupt": [], "ramped": []} for d in RD}; fpr_g = {d: [] for d in RD}
    for b in TW:
        ref_w = list(range(b - G - LAGB, b - G))
        rows_c = []
        for v in ref_w:
            j = b - v; vol = int(ramp_vol(j, Lr))
            if vol > 0: rows_c.append(np.random.default_rng(v).choice(ITEM_MM, vol, replace=False))
        n_ref = sum(len(np.concatenate(GAM[v])) for v in ref_w if v >= 0 and GAM[v]); sc_ = (100000 if not (QUICK or SYNTHETIC_TEST) else 5000) / max(n_ref, 1)
        extra = np.concatenate([c[: int(len(c) * sc_)] for c in rows_c]) if rows_c else None
        Bt, Bg = bank_from(ref_w, extra), bank_from(ref_w)                                  # contaminated and clean guard-band references
        hist = []
        for v in [v for v in NUL if 8 <= v < b - G]:
            j = b - v; vol = int(ramp_vol(j, Lr))
            if vol >= 50:
                cw = []
                for k in range(R["T_CAL"]):
                    r = np.random.default_rng(SEED + 9_700 + 100 * v + k); cv = vols_for(v, r, K_COLL, vol)
                    out = attacked(v, "dataset", r, r, banks=Bt, reps=["minilm"], camp=split_by(r.choice(ITEM_MM, int(cv.sum()), replace=False), cv))
                    if out: cw.append(out[0])
                hist += [cw[i % len(cw)] for i in range(len(NUL[v]))] if cw else NUL[v]
            else: hist += NUL[v]
        Tr = {d: float(np.quantile([s[d] for s in hist], 1 - ALPHA)) for d in RD}
        clean = [s for v in NUL if 8 <= v < b - G for s in NUL[v]]; Tg = {d: float(np.quantile([s[d] for s in clean], 1 - ALPHA)) for d in RD}
        for rep in range(R["T_REPS"]):
            r = np.random.default_rng(SEED + 9_800 + 100 * b + rep); st = r.bit_generator.state
            cv = vols_for(b, np.random.default_rng(rep), K_COLL, N_FIX); camp = split_by(np.random.default_rng(rep + 50).choice(ITEM_MM, int(cv.sum()), replace=False), cv)
            out_g = attacked(b, None, r, None, banks=Bg, reps=["minilm"], camp=camp); r2 = np.random.default_rng(0); r2.bit_generator.state = st
            out_r = attacked(b, None, r2, None, banks=Bt, reps=["minilm"], camp=camp)
            if out_g is None or out_r is None: continue
            for d in RD:
                det[d]["abrupt"].append(float(out_g[0][d] > Tg[d] and not out_g[1][d] > Tg[d])); det[d]["ramped"].append(float(out_r[0][d] > Tr[d] and not out_r[1][d] > Tr[d]))
        for t in range(min(10, len(NUL[b]))):
            i, l, _ = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); s = evaluate(i, l, b, Bg, ["minilm"])
            for d in RD: fpr_g[d].append(float(s[d] > Tg[d]))
    RAMP[(Lr, G)] = {d: dict(abrupt=float(np.mean(det[d]["abrupt"])) if det[d]["abrupt"] else float("nan"), ramped=float(np.mean(det[d]["ramped"])) if det[d]["ramped"] else float("nan"), fpr_guard=float(np.mean(fpr_g[d]))) for d in RD}
    print(f"[ramp L={Lr} guard={G}] " + " | ".join(f"{d.split('_')[0]} abrupt {RAMP[(Lr, G)][d]['abrupt']:.2f} ramped {RAMP[(Lr, G)][d]['ramped']:.2f} FPR(guard ref) {RAMP[(Lr, G)][d]['fpr_guard']:.3f}" for d in RD) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# # Predictions and VERDICT REPORT — copy into the chat

# %%
# ===== Cell 14: predictions and verdict =====
g = lambda src, d: RES[src][d]["mean"] if src in RES and d in RES[src] else float("nan")
au = lambda src, d: RES[src][d]["auc"] if src in RES and d in RES[src] else float("nan")
calib = ["tsC_minilm", "tsRc_minilm", "Tmax_minilm", "tsRc_lexical", "tsRc_style", "topic", "disc"]
S = lambda src, d: SEQ.get((src, d), {})
print(f"P36a natural_users: hollowed AUC {au('natural_users','tsRc_minilm'):.2f} (<= 0.65); max deployed {max(g('natural_users', d) for d in calib):.2f} (<= 0.05); old random_draws AUC {au('random_draws','tsRc_minilm'):.2f}")
print(f"P36b tilted_users: max deployed {max(g('tilted_users', d) for d in calib):.2f} (<= 0.05); hollowed martingale alarm within {R['SEQ_H']} weeks {S('tilted_users','tsRc_minilm').get('m_frac', float('nan')):.2f} of runs (<= 0.10)")
print(f"P36c tilted_c05 (old injection): hollowed martingale alarm {S('tilted_c05','tsRc_minilm').get('m_frac', float('nan')):.2f} of runs (>= 0.5); median {S('tilted_c05','tsRc_minilm').get('m_med')}")
print(f"P36d padded: hollowed martingale median {S('padded','tsRc_minilm').get('m_med')} weeks (<= 6); first weekly alarm median {S('padded','tsRc_minilm').get('f_med')}")
print(f"P36e written_c05: hollowed {g('written_c05','tsRc_minilm'):.2f} (<= 0.10); discriminator {g('written_c05','disc'):.2f} (>= 0.5) at FPR {FPR['disc'][0]:.4f} (<= 0.02); written_users: hollowed {g('written_users','tsRc_minilm'):.2f}, discriminator {g('written_users','disc'):.2f}")
print(f"P36f relay_mix: embedding max {max(g('relay_mix', d) for d in ['tsC_minilm','tsRc_minilm','Tmax_minilm']):.2f} (<= 0.05), topicality {g('relay_mix','topic'):.2f} (>= 0.3); relay_tilt hollowed {g('relay_tilt','tsRc_minilm'):.2f} (>= 0.3)")
print(f"P36g lexical hollowed: dataset {g('dataset','tsRc_lexical'):.2f} (>= 0.8), padded {g('padded','tsRc_lexical'):.2f} (>= 0.3); style hollowed dataset {g('dataset','tsRc_style'):.2f} (>= 0.5); FPR lexical {FPR['tsRc_lexical'][0]:.4f}, style {FPR['tsRc_style'][0]:.4f}")
print(f"P36h MinHash: FPR {FPR['dup'][0]:.4f} (<= 0.02); dataset {g('dataset','dup'):.2f}, sweep {g('eval_gsm8k','dup'):.2f} (<= 0.1)")
r8, rg8, rg16 = RAMP[(8, 0)], RAMP[(8, R['GUARD'])], RAMP[(16, R['GUARD'])]
print(f"P36i ramp, hollowed: no guard {r8['tsRc_minilm']['abrupt']:.2f}->{r8['tsRc_minilm']['ramped']:.2f}; guard {R['GUARD']} vs 8-week ramp {rg8['tsRc_minilm']['abrupt']:.2f}->{rg8['tsRc_minilm']['ramped']:.2f} (>= 0.8x) at FPR {rg8['tsRc_minilm']['fpr_guard']:.3f} (<= 0.02); 16-week ramp {rg16['tsRc_minilm']['ramped']:.2f} (drop >= 0.3)")
for src in ["dataset", "padded"]:
    print(f"P36j {src} hollowed: week CI {RES[src]['tsRc_minilm']['week_ci']}, hierarchical CI {RES[src]['tsRc_minilm']['h_ci']}; by window size (large/mid/small A; colluder share {[round(x, 2) for x in RES[src]['share_terciles']]}): {RES[src]['strata']['tsRc_minilm']}")
print(f"P36k FPR topicality {FPR['topic'][0]:.4f}, discriminator {FPR['disc'][0]:.4f} (<= 0.02)")
print("=" * 90); print("STEP 36 VERDICT REPORT"); print("=" * 90)
print("[FPR]", json.dumps({d: [round(v, 4) for v in FPR[d]] for d in DETS}))
for src, res in RES.items():
    print(f"[W] {src:>14} " + " ".join(f"{d}={res[d]['mean']:.2f}{[round(x, 2) for x in res[d]['h_ci']]} o={res[d]['tpr1']:.2f} auc={res[d]['auc']:.2f}" for d in DETS))
    if "strata" in res: print(f"    strata (large/mid/small A) {json.dumps(res['strata'])} shares {res['share_terciles']}")
print("[SEQ]", json.dumps({f"{k[0]}|{k[1]}": v for k, v in SEQ.items()}))
print("[RAMP]", json.dumps({f"L{k[0]}_G{k[1]}": v for k, v in RAMP.items()}))
print(f"[W-AUC] attacker discriminator {RW_AUC}; passing rewrites {len(RW_PASS)}; task share {P_TASK:.4f}")
print("=" * 90)
