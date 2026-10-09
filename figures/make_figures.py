"""Draws every data figure in "The Limits of Query Monitoring for Distributed Model Extraction".

The figures plot numbers recorded from the experiment scripts (Kaggle runs); this script does not
re-run experiments. Each data array below is annotated with the script that produces it, so a reader can
re-run that script and compare.  Output: figures/out/*.pdf   Usage: python figures/make_figures.py

  Paper figure                                  data source (experiments/...)
  Fig. 1a  capability vs hollowed detection    x: 04_frozen_holdout/step35 (dataset, padding, tilted);
                                               03_capability_frontier/step32 (two written campaigns, dev. weeks)
                                               y: 03_capability_frontier/step19 (dataset, padding), step29d (tilted),
                                               step32 (written campaigns)
  Fig. 1b  frozen detectors, untouched weeks   04_frozen_holdout/step35 (Table 3, MiniLM 64-252 accounts)
  Fig. 2   monitor pipeline                    diagram, no data
  Fig. 3   barrier in a controlled setting     01_controlled_studies/step08 (App. E, Table 4)
  Fig. 4a  calibration schemes                 02_development_weeks/step10 (App. Table 5, last row)
  Fig. 4b  detector false alarms               02_development_weeks/step12 (Table 1, "Realised false alarms")
  Fig. 5   scale / selection gap               02_development_weeks/step20 (top eigenvector, true colluders; App. Table 10)
                                               02_development_weeks/step27_31 (every outlier eigenvector; App. Table 11)
  Fig. 6a  extraction regime (SmolLM2-360M)    03_capability_frontier/step19 (App. Table 14), step29d (tilted)
  Fig. 6b  elicitation regime (Qwen2.5-0.5B)   03_capability_frontier/step16 (App. Table 13)
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9.5,
                     "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "dejavusans", "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
C = dict(blue="#1f6fb4", orange="#d97e00", green="#2a9d6f", red="#c0392b", purple="#7b4fa0", grey="#808080", dark="#333333")
W = 6.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

# ---------- Fig. 1: the paper at a glance ----------
# (label, hollowed weekly detection, mean GSM8K gain, [seed range], colour, marker)
# [PAPER] Fig. 1a x-values 1.00 / 0.57 / 0.04: Table 3, "Hollowed" column -> experiments/04_frozen_holdout/step35
# [PAPER] Fig. 1a x-values 0.00 / 0.05 for written campaigns (development weeks) -> experiments/03_capability_frontier/step32
# [PAPER] Fig. 1a y-values 10.4 / 7.6 (3k queries, two seeds): App. Table 14 -> experiments/03_capability_frontier/step19
# [PAPER] Fig. 1a y-values 4.4 (distribution-matched), written in users' voice -> step32; 2.2 (tilted) -> step29d
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1.08, 1], wspace=0.32))
camps = [("dataset prompts", 1.00, 10.4, [10.25, 10.5], C["red"], "o"),
         ("half padding", 0.57, 7.6, [7.0, 8.25], C["purple"], "s"),
         ("written, distribution-matched", 0.00, 4.4, [4.0, 4.8], C["green"], "D"),
         ("written in users' voice", 0.05, 3.7, [3.7, 3.7], C["dark"], "^"),
         ("least detectable (tilted)", 0.04, 2.2, [2.0, 2.5], C["orange"], "v")]
ax[0].axvspan(-0.06, 0.12, color="#ececec", zorder=0)
ax[0].text(0.14, 13.1, "\u2190 shaded: evades detection", fontsize=8, color="#444444", ha="left", va="top")
for lab, x, m, g, c_, mk in camps:
    ax[0].errorbar([x], [m], yerr=[[m - min(g)], [max(g) - m]], fmt=mk, color=c_, ms=6.5, capsize=2.5, lw=1.1, label=lab)
ax[0].set_xlim(-0.06, 1.06); ax[0].set_ylim(0, 13.5)
ax[0].set_xlabel("weekly detection (hollowed selection)"); ax[0].set_ylabel("capability gained\n(GSM8K points)")
ax[0].set_title("(a) What an attacker gets", loc="left")
ax[0].legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=7.6, handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
dets = ["Two-scale", "Hollowed", "Multi-outlier"]
# [PAPER] Fig. 1b = Table 3 (MiniLM, 64-252 accounts): dataset / half padding / least detectable rows -> step35
vals = {"dataset prompts": [0.86, 1.00, 0.97], "half padding": [0.00, 0.57, 0.00], "least detectable": [0.00, 0.04, 0.00]}
cols = [C["red"], C["purple"], C["orange"]]; x = np.arange(3); w = 0.29
for i, (k, v) in enumerate(vals.items()):
    bars = ax[1].bar(x + (i - 1) * w, v, w, color=cols[i], label=k)
    for xb, vb in zip(x + (i - 1) * w, v):
        if vb >= 0.01: ax[1].text(xb, vb + 0.02, f"{vb:.2f}", ha="center", va="bottom", fontsize=7)
# [PAPER] Fig. 1b dotted line: realised false alarms 1.9% (Table 3, "False alarms", Hollowed) -> step35
ax[1].axhline(0.019, color="k", lw=0.8, ls=":", label="false alarms (1.9%)")
ax[1].set_xticks(x); ax[1].set_xticklabels(dets, fontsize=8.5); ax[1].set_xlim(-0.55, 2.55); ax[1].set_ylim(0, 1.12); ax[1].set_ylabel("weekly detection")
ax[1].set_title("(b) Frozen detectors, 17 untouched weeks", loc="left")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=8, columnspacing=1.0, handlelength=1.4)
plt.savefig(f"{OUT}/glance.pdf"); plt.close()

# ---------- Fig. 2: the monitor (diagram, §3) ----------
fig, ax = plt.subplots(figsize=(W, 2.55)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, txt, fc, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="#666666", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.3)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#444444"))
steps = [("1  Queries\nof every\naccount", "#e8eef7"), ("2  Topic\nfingerprint\nper account", "#e9f5ee"),
         ("3  Compare\nwith last\nmonth", "#fdf1dc"), ("4  Hollow:\nkeep only\nco-movement", "#f6e6ef"),
         ("5  Most\ncoherent\ngroup", "#ece8f5")]
bw, gap = 16.4, 4.5
for i, (t, fc) in enumerate(steps):
    x0 = i * (bw + gap); box(x0 + 0.4, 23, bw, 15, t, fc)
    if i < 4: arrow(x0 + bw + 1.0, 30.5, x0 + bw + gap - 0.2, 30.5)
box(0.4, 1, 46, 15, "Calibration: thresholds come from whole\npast weeks, never from shuffled accounts,\nso benign drift and coordination are\nin the null by construction.", "#f2f2f2", 7.8)
box(52.6, 1, 46.8, 15, "6  Score the group (two-scale score) and\nfire if it beats every past benign week.\nOutput: a short list of accounts\nfor human review.", "#eef6ee", 7.8)
arrow(4 * (bw + gap) + bw / 2 + 0.4, 22.6, 80, 16.6)
plt.savefig(f"{OUT}/monitor.pdf"); plt.close()

# ---------- Fig. 3: the barrier in a controlled setting (App. B) ----------
# [PAPER] Fig. 3 power vs strength s, 100 and 1,000 accounts -> experiments/01_controlled_studies/step08 (App. E, Table 4)
fig, ax = plt.subplots(figsize=(3.6, 2.5))
s = ["0", "0.05", "0.1", "0.25", "0.5", "1"]; xs = np.arange(6)
ax.plot(xs, [1, 1, 1, 0, 0, 0], "o-", color=C["blue"], ms=5, label="100 accounts")
ax.plot(xs, [1, .67, .04, 0, .04, .04], "s-", color=C["orange"], ms=5, label="1,000 accounts")
ax.set_xticks(xs); ax.set_xticklabels(s); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("strength of benign coordination"); ax.set_ylabel("detection power")
ax.legend(loc="upper right", frameon=False)
plt.savefig(f"{OUT}/barrier_controlled.pdf"); plt.close()

# ---------- Fig. 4: calibration (App. D.1) ----------
# [PAPER] Fig. 4a held-out false alarms 18.8/5.0/3.4% (100 accounts), 56.2/24.7/28.7% (500 accounts):
#         App. Table 5 last row -> experiments/02_development_weeks/step10
fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.3))
sch = ["fixed", "rolling", "rolling +\nlagged ref."]; v100 = [18.8, 5.0, 3.4]; v500 = [56.2, 24.7, 28.7]; xs = np.arange(3); w = 0.36
ax[0].bar(xs - w / 2, v100, w, color=C["blue"], label="100 accounts"); ax[0].bar(xs + w / 2, v500, w, color=C["orange"], label="500 accounts")
for i in range(3):
    ax[0].text(xs[i] - w / 2, max(v100[i] + 1.2, 6.5), f"{v100[i]:.1f}", ha="center", fontsize=7.5)
    ax[0].text(xs[i] + w / 2, max(v500[i] + 1.2, 6.5), f"{v500[i]:.1f}", ha="center", fontsize=7.5)
ax[0].axhline(5, color="k", ls=":", lw=0.9, label="nominal 5%")
ax[0].set_xticks(xs); ax[0].set_xticklabels(sch, fontsize=8.5); ax[0].set_ylabel("false alarms, held-out (%)"); ax[0].set_ylim(0, 66)
ax[0].set_title("(a) Calibration schemes", loc="left"); ax[0].legend(fontsize=8, frameon=False, loc="upper right")
# [PAPER] Fig. 4b means = Table 1 "Realised false alarms" row (0.7 ... 14.2%), bars = week-clustered 95% intervals
#         -> experiments/02_development_weeks/step12
dets = ["two-\nscale", "Fraudar", "global", "per-\naccount", "HC", "spectral", "MMD", "volume"]
m = [0.7, 2.9, 4.1, 5.3, 7.5, 7.8, 11.6, 14.2]; lo = [0, 0, 0, 0, 0, 0, 2.7, 3.5]; hi = [2.2, 8.7, 10.0, 13.6, 16.3, 16.6, 22.6, 26.9]
cols = [C["green"]] + [C["grey"]] * 7
ax[1].bar(range(8), m, color=cols, yerr=[np.array(m) - lo, np.array(hi) - m], capsize=2.5, error_kw=dict(lw=0.8))
ax[1].axhline(1, color="k", ls=":", lw=0.9, label="nominal 1%")
ax[1].set_xticks(range(8)); ax[1].set_xticklabels([d.replace("\n", " ").replace("- ", "-") for d in dets], fontsize=8, rotation=35, ha="right", rotation_mode="anchor"); ax[1].set_ylabel("false alarms, held-out (%)")
ax[1].set_title("(b) Detectors, whole weekly populations", loc="left"); ax[1].legend(fontsize=8, frameon=False, loc="upper left")
plt.savefig(f"{OUT}/calibration.pdf"); plt.close()

# ---------- Fig. 5: scale, oracle thresholds (App. D.5) ----------
# [PAPER] Fig. 5 "top eigenvector only" 1.00/0.80/0.42/0.15/0.00 and "true colluders scored": App. Table 10
#         -> experiments/02_development_weeks/step20
# [PAPER] Fig. 5 "every outlier eigenvector" 1.00/1.00/1.00/0.93/1.00: App. Table 11
#         -> experiments/02_development_weeks/step27_31
fig, ax = plt.subplots(figsize=(4.6, 2.9))
A = [250, 1000, 2500, 5000, 10000]; share = ["20%", "5%", "2%", "1%", "0.5%"]
ax.plot(A, [1, 1, 1, 1, 1], "s--", color=C["green"], ms=5, label="true colluders scored (oracle)")
ax.plot(A, [1.00, 1.00, 1.00, 0.93, 1.00], "^-", color=C["purple"], ms=5, label="every outlier eigenvector")
ax.plot(A, [1.00, 0.80, 0.42, 0.15, 0.00], "o-", color=C["blue"], ms=5, label="top eigenvector only")
ax.set_xscale("log"); ax.set_xticks(A); ax.set_xticklabels([f"{a:,}\n{s_}" for a, s_ in zip(A, share)], fontsize=8.5)
ax.minorticks_off(); ax.set_ylim(-0.05, 1.1); ax.set_ylabel("detection at 1% false alarms\n(oracle threshold)")
ax.set_xlabel("accounts monitored, colluder share")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)
plt.savefig(f"{OUT}/scale.pdf"); plt.close()

# ---------- Fig. 6: price of stealth by regime (App. G) ----------
# [PAPER] Fig. 6a all-math / half-padding gains per seed at 1k/3k/10k queries (means 7.5/10.4/15.9; padding costs
#         3.6/2.8/3.8): App. Table 14 -> experiments/03_capability_frontier/step19
# [PAPER] Fig. 6a tilted campaign (+2.2 at 3k, +1.5 at 10k) -> experiments/03_capability_frontier/step29d
# [PAPER] Fig. 6b Qwen2.5-0.5B, mean +- s.d. over three seeds: App. Table 13 -> experiments/03_capability_frontier/step16
fig, ax = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw=dict(wspace=0.3))
B = [1000, 3000, 10000]
U = {1000: [7.5, 7.5], 3000: [10.5, 10.25], 10000: [16.75, 15.0]}; M = {1000: [4.5, 3.25], 3000: [8.25, 7.0], 10000: [13.25, 11.0]}
for d, c_, mk, lab in [(U, C["red"], "o", "all math"), (M, C["purple"], "s", "half natural padding")]:
    mm = [np.mean(d[b]) for b in B]
    ax[0].errorbar(B, mm, yerr=[[np.mean(d[b]) - min(d[b]) for b in B], [max(d[b]) - np.mean(d[b]) for b in B]], fmt=mk + "-", color=c_, capsize=3, ms=5, lw=1.3, label=lab)
T3, T10 = [2.0, 2.5], [1.7, 1.2]
ax[0].errorbar([3000, 10000], [np.mean(T3), np.mean(T10)], yerr=[[np.mean(T3) - min(T3), np.mean(T10) - min(T10)], [max(T3) - np.mean(T3), max(T10) - np.mean(T10)]],
               fmt="v--", color=C["orange"], capsize=3, ms=5, lw=1.3, label="least detectable (tilted)")
ax[0].set_xscale("log"); ax[0].set_xticks(B); ax[0].set_xticklabels(["1k", "3k", "10k"]); ax[0].minorticks_off()
ax[0].set_xlabel("campaign queries"); ax[0].set_ylabel("GSM8K gain (points)"); ax[0].set_ylim(0, 19)
ax[0].set_title("(a) Extraction: SmolLM2-360M", loc="left"); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
B2 = [100, 300, 900]
ax[1].errorbar([b / 1.13 for b in B2], [9.8, 7.5, 8.6], yerr=[2.1, 0.5, 1.2], fmt="o-", color=C["red"], capsize=3, ms=5, lw=1.3, label="all math")
ax[1].errorbar([b * 1.13 for b in B2], [9.8, 10.2, 10.1], yerr=[1.8, 2.1, 1.0], fmt="s-", color=C["purple"], capsize=3, ms=5, lw=1.3, label="half natural padding")
ax[1].set_xscale("log"); ax[1].set_xticks(B2); ax[1].set_xticklabels(["100", "300", "900"]); ax[1].minorticks_off()
ax[1].set_xlabel("campaign queries"); ax[1].set_ylabel("GSM8K gain (points)"); ax[1].set_ylim(0, 19)
ax[1].set_title("(b) Elicitation: Qwen2.5-0.5B", loc="left"); ax[1].legend(fontsize=8, frameon=False, loc="upper left")
plt.savefig(f"{OUT}/price_of_stealth.pdf"); plt.close()
print("figures written")
