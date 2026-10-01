"""Phase 5 figures. Reads saved results only; never re-scores the test set.

Inputs (all already on disk):
  reports/phase5/test_results.json        saved by src/evaluate_test.py --stage test (used once)
  reports/phase5/test_predictions.parquet saved scores, labels and line payments for the test split
  reports/shap_fit.json                   SHAP on a 20,000-row val sample (Phase 2)
  reports/phase4/runs/*/summary.json      grounding results (Phase 4)
Outputs: reports/phase5/figures/*.png and figure_checks.txt (each recomputed number vs the saved one).

Run: python scripts/make_phase5_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "reports" / "phase5"
OUT = P5 / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette: first three categorical slots of the validated default (they pass all-pairs checks).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TYPE_COLOR = {"carrier": BLUE, "outpatient": ORANGE, "dme": AQUA}
TYPE_LABEL = {"carrier": "Carrier", "outpatient": "Outpatient", "dme": "DME"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200, "savefig.facecolor": "white",
})

RES = json.load(open(P5 / "test_results.json"))
PROTO = RES["protocol"]
R = PROTO["headline"]["r"]
C = PROTO["headline"]["c"]
WAGE, B = 37.51, PROTO["headline"]["b"]
pred = pd.read_parquet(P5 / "test_predictions.parquet")
p, y, A = pred["p"].to_numpy(), pred["y"].to_numpy(), pred["amount"].to_numpy()
CHECKS = []


def check(name, mine, saved, tol=1e-6):
    ok = abs(mine - saved) <= tol * max(1.0, abs(saved))
    CHECKS.append(f"{'OK  ' if ok else 'FAIL'} {name}: recomputed {mine:.6f} vs saved {saved:.6f}")
    assert ok, CHECKS[-1]


def loss_if_missed(amount, rho):
    return amount * (1 - rho) + C * rho * amount


def review_cost(m):
    return WAGE * B * m / 60.0


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 1: ranking for scarce reviewers --------------------------------------------
def fig_ranking():
    rho = 0.24
    L = loss_if_missed(A, rho)
    n = len(p)
    ks = np.unique(np.concatenate([np.linspace(0.005, 0.30, 60), [0.01, 0.02, 0.05, 0.10, 0.20]]))
    orders = {"By score p": np.argsort(-p, kind="stable"), "By value p*L - r": np.argsort(-(p * L - R), kind="stable")}
    denied_dollars = (y * L).sum()
    curves = {}
    for name, order in orders.items():
        cy = np.cumsum(y[order])
        cd = np.cumsum((y * L)[order])
        idx = np.round(ks * n).astype(int)
        curves[name] = (cy[idx - 1] / idx, cd[idx - 1] / denied_dollars)
    # check against the saved top-k table (dollars share there is of dollars on denied lines at this rho)
    saved = {r_["k_frac"]: r_ for r_ in RES["scored"]["overall"]["topk"]["rho_0.24"]}
    for kf, rec in saved.items():
        i = int(np.where(np.isclose(ks, kf))[0][0])
        check(f"precision@{kf:.0%} by score", curves["By score p"][0][i], rec["by_score"]["precision"])
        check(f"precision@{kf:.0%} by value", curves["By value p*L - r"][0][i], rec["by_value"]["precision"])
        check(f"dollar share@{kf:.0%} by score", curves["By score p"][1][i], rec["by_score"]["dollars_captured_share"], 1e-5)
        check(f"dollar share@{kf:.0%} by value", curves["By value p*L - r"][1][i], rec["by_value"]["dollars_captured_share"], 1e-5)
    base = y.mean()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    cols = {"By score p": BLUE, "By value p*L - r": ORANGE}
    for name, (prec, share) in curves.items():
        ax[0].plot(ks * 100, prec, color=cols[name], lw=2, label=name)
        ax[1].plot(ks * 100, share * 100, color=cols[name], lw=2, label=name)
    ax[0].axhline(base, color=MUTED, lw=1, ls="--")
    ax[0].text(29.5, base + 0.02, f"all lines: {base:.1%}", ha="right", color=MUTED, fontsize=8)
    ax[0].set_title("Share of reviewed lines that are denials", loc="left", fontsize=10)
    ax[0].set_ylabel("Precision"); ax[0].set_ylim(0, 1.02)
    ax[1].set_title("Share of denied dollars covered", loc="left", fontsize=10)
    ax[1].set_ylabel("% of denied dollars (loss L)"); ax[1].set_ylim(0, 102)
    for a in ax:
        a.set_xlabel("Lines reviewed (% of all lines, highest first)")
    ax[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    fig.suptitle("Review the highest scores to find denials; review the highest value to find dollars (test, rho = 0.24)",
                 x=0.01, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig1_ranking_precision_and_dollars.png")


# ---- Figure 2: calibration -------------------------------------------------------------
def fig_calibration():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.6, 6.6), gridspec_kw={"height_ratios": [4, 1.2], "hspace": 0.28})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.66, 0.57, "perfect", color=MUTED, fontsize=8, rotation=38)
    groups = [("overall", "Overall", INK)] + [(k, TYPE_LABEL[k], TYPE_COLOR[k]) for k in TYPE_COLOR]
    for key, label, col in groups:
        d = RES["scored"][key]
        cal = [b for b in d["calibration"] if b["n"] > 0]
        xs = [b["mean_score"] for b in cal]; ys = [b["observed_rate"] for b in cal]
        ax.plot(xs, ys, color=col, lw=1.6 if key != "overall" else 2.2, marker="o", ms=4,
                label=f"{label} (ECE {d['ece']:.3f})", zorder=3 if key == "overall" else 2)
    ax.set_xlabel("Mean predicted score in bin"); ax.set_ylabel("Observed denial rate in bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Close to the diagonal where most lines sit; the few bins from 0.6 to\n0.9 run above it (more denials than the score says)",
                 loc="left", fontsize=9.5)
    cal = RES["scored"]["overall"]["calibration"]
    ax2.bar([(b["bin_lo"] + b["bin_hi"]) / 2 for b in cal], [max(b["n"], 0.5) for b in cal], width=0.085, color=MUTED)
    ax2.set_yscale("log"); ax2.set_xlim(0, 1); ax2.set_ylabel("Lines (log)")
    ax2.set_xlabel("Score bin (test, 10 equal-width bins)")
    save(fig, "fig2_calibration.png")


# ---- Figure 3: cost versus a single cutoff, with cost-ratio sensitivity ----------------
def net_at_cutoff(t, rho, r):
    flag = p > t
    L = loss_if_missed(A, rho)
    return (y[flag] * L[flag]).sum() - r * flag.sum()


def fig_cost_threshold():
    ts = np.linspace(0.0, 0.9, 181)
    ms = [(10, "#1baf7a"), (20, BLUE), (30, ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=False)
    sweep = {(round(s["rho"], 2), s["m"], round(s["b"], 2), round(s["c"], 3)): s for s in RES["scored"]["overall"]["sweep"]}
    for ax, rho in zip(axes, (0.24, 0.55)):
        for m, col in ms:
            r = review_cost(m)
            L = loss_if_missed(A, rho)
            vals = np.array([net_at_cutoff(t, rho, r) for t in ts]) / 1e6
            ax.plot(ts, vals, color=col, lw=1.8, label=f"cutoff t, m = {m} min (r = ${r:.2f})")
            best = int(vals.argmax())
            ax.plot(ts[best], vals[best], "o", color=col, ms=5)
            # per-line rule: flag p*L > r (needs no cutoff search)
            flag = p * L > r
            rule = ((y * L)[flag].sum() - r * flag.sum()) / 1e6
            ax.axhline(rule, color=col, lw=1, ls=":")
            s = sweep[(rho, float(m), B, C)]
            check(f"rule net savings rho={rho} m={m}", rule * 1e6, s["net_savings"], 1e-6)
            if m == 20:
                review_all = ((y * L).sum() - r * len(p)) / 1e6
                check(f"review-all net rho={rho}", review_all * 1e6, s["review_all_net_savings"], 1e-6)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f"rho = {rho}", loc="left", fontsize=10)
        ax.set_xlabel("One cutoff t for every line: review when p > t")
        ax.set_ylabel("Net savings vs reviewing nothing ($M)")
    axes[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=8)
    fig.text(0.01, -0.04, "Solid lines: one cutoff t for every line; dots mark the best t picked with hindsight on test, an upper bound for any single cutoff. "
             "Dotted lines: the per-line rule p*L > r, which needs no cutoff search.",
             fontsize=8, color=MUTED, ha="left")
    fig.suptitle("The per-line rule beats even the hindsight-best single cutoff at every review time tested (m = 10, 20, 30 min)",
                 x=0.01, ha="left", fontsize=10.5, y=1.03)
    save(fig, "fig3_cost_vs_cutoff.png")


# ---- Figure 4: decision curve ----------------------------------------------------------
def fig_decision_curve():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    for ax, key in zip(axes.ravel(), ["overall", "carrier", "outpatient", "dme"]):
        dc = RES["scored"][key]["decision_curve"]
        t = [d["t"] for d in dc]
        col = INK if key == "overall" else TYPE_COLOR[key]
        ax.plot(t, [d["model"] for d in dc], color=col, lw=2, label="Model")
        ax.plot(t, [d["review_all"] for d in dc], color=MUTED, lw=1.4, ls="--", label="Review all")
        ax.axhline(0, color=MUTED, lw=1, label="Review none")
        ax.set_title("Overall" if key == "overall" else TYPE_LABEL[key], loc="left", fontsize=10)
        lo = min(min(d["model"] for d in dc), -0.01)
        ax.set_ylim(lo, max(d["model"] for d in dc) * 1.1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Threshold probability t")
    for ax in axes[:, 0]:
        ax.set_ylabel("Net benefit (per line)")
    fig.suptitle("Decision curve: net benefit of acting on the model versus reviewing everything or nothing (test)",
                 x=0.01, ha="left", fontsize=10.5, y=0.98)
    save(fig, "fig4_decision_curve.png")


# ---- Figure 5: SHAP importance ---------------------------------------------------------
def fig_shap():
    s = json.load(open(ROOT / "reports" / "shap_fit.json"))
    top = list(s["mean_abs_shap_overall_top20"].items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.barh([k for k, _ in top], [v for _, v in top], color=BLUE, height=0.65)
    ax.grid(axis="y", visible=False)
    for i, (_, v) in enumerate(top):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Mean |SHAP| (log-odds units)")
    ax.set_title("HCPCS procedure code dominates the model's scores", loc="left", fontsize=10)
    fig.text(0.01, -0.03, "20,000-row label-stratified validation sample, model v2. SHAP was not recomputed on test.",
             fontsize=8, color=MUTED)
    save(fig, "fig5_shap_importance.png")


# ---- Figure 6: grounding results by kind of token --------------------------------------
RUNS = [
    ("20260928-222319-final-v1", "v1 prompt,\noriginal tools"),
    ("20260928-222954-final-v1-newtools", "v1 prompt,\nprecomputed numbers"),
    ("20260928-223448-final-v2-newtools", "v2 prompt\n(30 claims)"),
    ("20260928-224007-heldout-v2", "v2 prompt\n(30 held out)"),
    ("20260929-174122-heldout-v2-defs", "v2 + sourced\ndefinitions"),
]
KINDS = [("number", "Numbers (model arithmetic)", BLUE), ("code", "Codes", ORANGE), ("carc", "Denial reason codes", AQUA), ("desc", "Descriptions (code meanings from memory)", "#eda100")]


def fig_grounding():
    rows = []
    for d, label in RUNS:
        s = json.load(open(ROOT / "reports" / "phase4" / "runs" / d / "summary.json"))["pooled"]
        row = {"label": label, "total": s["tokens"], "grounded": s["grounded_tokens"], "empty": s["empty_explanations"]}
        for k, _, _ in KINDS:
            bk = s["by_kind"].get(k, {"tokens": 0, "grounded": 0})
            row[k] = bk["tokens"] - bk["grounded"]
        assert sum(row[k] for k, _, _ in KINDS) == row["total"] - row["grounded"], d
        rows.append(row)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(df)); bottom = np.zeros(len(df))
    for k, label, col in KINDS:
        ax.bar(x, df[k], bottom=bottom, color=col, width=0.55, label=label, edgecolor="white", linewidth=1.5)
        bottom += df[k].to_numpy()
    for i, r_ in df.iterrows():
        ax.text(i, bottom[i] + 0.25, f"{r_['grounded']}/{r_['total']} grounded\n{r_['grounded'] / r_['total']:.1%}"
                + (f"\n{r_['empty']} empty answers" if r_["empty"] else ""), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("Checkable tokens NOT traceable to a tool output")
    ax.set_ylim(0, max(bottom.max() + 4, 6))
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Ungrounded tokens by kind, per run (30 claims each)", loc="left", fontsize=10)
    fig.text(0.01, -0.03, "Grounded = traceable to a tool output, not necessarily correct. "
             "Policy over-statements are not counted here (found by reading, not measured).", fontsize=8, color=MUTED)
    save(fig, "fig6_grounding_by_failure_type.png")
    return df


if __name__ == "__main__":
    fig_ranking(); fig_calibration(); fig_cost_threshold(); fig_decision_curve(); fig_shap(); df = fig_grounding()
    (OUT / "figure_checks.txt").write_text("\n".join(CHECKS) + "\n")
    print("\n".join(CHECKS)); print(df.to_string())
