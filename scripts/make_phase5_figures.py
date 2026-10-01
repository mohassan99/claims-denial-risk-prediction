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


# ---- Figure 1: which lines to review first ---------------------------------------------
def fig_ranking():
    rho = 0.24
    L = loss_if_missed(A, rho)
    n = len(p)
    ks = np.unique(np.concatenate([np.linspace(0.005, 0.30, 60), [0.01, 0.02, 0.05, 0.10, 0.20]]))
    by_score = np.argsort(-p, kind="stable")
    by_value = np.argsort(-(p * L - R), kind="stable")
    denied_dollars = (y * L).sum()
    idx = np.round(ks * n).astype(int)
    curves = {}
    for name, order in (("score", by_score), ("value", by_value)):
        cy, cd = np.cumsum(y[order]), np.cumsum((y * L)[order])
        curves[name] = (cy[idx - 1] / idx * 100, cd[idx - 1] / denied_dollars * 100)
    saved = {r_["k_frac"]: r_ for r_ in RES["scored"]["overall"]["topk"]["rho_0.24"]}
    for kf, rec in saved.items():
        i = int(np.where(np.isclose(ks, kf))[0][0])
        check(f"precision@{kf:.0%} by score", curves["score"][0][i] / 100, rec["by_score"]["precision"])
        check(f"precision@{kf:.0%} by value", curves["value"][0][i] / 100, rec["by_value"]["precision"])
        check(f"dollar share@{kf:.0%} by score", curves["score"][1][i] / 100, rec["by_score"]["dollars_captured_share"], 1e-5)
        check(f"dollar share@{kf:.0%} by value", curves["value"][1][i] / 100, rec["by_value"]["dollars_captured_share"], 1e-5)
    base = y.mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.6))
    names = {"score": ("Highest denial score first", BLUE), "value": ("Highest expected dollars first", ORANGE)}
    for key, (label, col) in names.items():
        ax[0].plot(ks * 100, curves[key][0], color=col, lw=2.2, label=label)
        ax[1].plot(ks * 100, curves[key][1], color=col, lw=2.2, label=label)
    ax[0].axhline(base, color=MUTED, lw=1, ls="--")
    ax[0].text(29.5, base + 2.5, f"Picking lines at random: {base:.0f}% are denials", ha="right", color=MUTED, fontsize=8.5)
    at = lambda key, which, kf: curves[key][which][int(np.where(np.isclose(ks, kf))[0][0])]
    for key, (label, col) in names.items():
        ax[0].plot(1, at(key, 0, .01), "o", color=col, ms=6, zorder=5)
        ax[1].plot(1, at(key, 1, .01), "o", color=col, ms=6, zorder=5)
    ax[0].text(8, 99, f"Dots: the top 1% of lines (3,600 lines)\nBy score: {at('score', 0, .01):.0f}% are denials\n"
               f"By dollars: {at('value', 0, .01):.0f}% are denials", va="top", fontsize=9, color=INK)
    ax[1].text(8, 46, f"Dots: the top 1% of lines (3,600 lines)\nBy dollars: covers {at('value', 1, .01):.0f}% of the money\n"
               f"By score: covers {at('score', 1, .01):.0f}% of the money", va="top", fontsize=9, color=INK)
    ax[0].set_title("Of the lines you review, how many are real denials?", loc="left", fontsize=10.5)
    ax[0].set_ylabel("% of reviewed lines that are denials"); ax[0].set_ylim(0, 102)
    ax[1].set_title("How much of the money at stake do you cover?", loc="left", fontsize=10.5)
    ax[1].set_ylabel("% of all dollars at stake (loss if a denial is missed)"); ax[1].set_ylim(0, 102)
    for a in ax:
        a.set_xlabel("Lines reviewed, as % of all lines")
    ax[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Review order depends on the goal: score order finds the most denials, dollar order finds the most money",
                 x=0.01, ha="left", fontsize=11.5, y=1.02)
    fig.text(0.01, -0.03, "Test set, 359,985 claim lines. Dollars at stake use a 24% overpayment recovery rate (hypothetical; the 55% case is in the results file). "
             "Expected dollars = chance of denial x loss if missed, minus the cost of a review.", fontsize=8, color=MUTED)
    save(fig, "fig1_ranking_precision_and_dollars.png")


# ---- Figure 2: calibration -------------------------------------------------------------
def fig_calibration():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 7.4), gridspec_kw={"height_ratios": [4, 1.3], "hspace": 0.38})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.30, 0.37, "model exactly right", color=MUTED, fontsize=8, rotation=38)
    # outpatient is the only type with lines scoring above 0.5, so draw overall first, outpatient last, on top
    order = [("overall", "All claim types", INK, 2.6), ("carrier", "Carrier (physician and supplier claims)", BLUE, 1.6),
             ("dme", "DME (equipment claims)", AQUA, 1.6), ("outpatient", "Outpatient (hospital outpatient claims)", ORANGE, 1.6)]
    for key, label, col, lw in order:
        d = RES["scored"][key]
        cal = [b for b in d["calibration"] if b["n"] > 0]
        ax.plot([b["mean_score"] for b in cal], [b["observed_rate"] for b in cal], color=col, lw=lw, marker="o", ms=4,
                label=f"{label}", zorder=2 if key == "overall" else 3)
    ax.annotate("Above 0.5 only outpatient lines\nappear (black and orange coincide):\nall but a handful are billed with a\nretired consultation code. Carrier\nand DME have no lines here.",
                xy=(0.76, 0.925), xytext=(0.50, 0.14), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.set_xlabel("What the model said: predicted chance of denial")
    ax.set_ylabel("What happened: share of those lines actually denied")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("When the model says 10% to 30%, about that share is denied.\nThe few lines it scores 60% to 90% are denied even more often.",
                 loc="left", fontsize=10)
    cal = RES["scored"]["overall"]["calibration"]
    ax2.bar([(b["bin_lo"] + b["bin_hi"]) / 2 for b in cal], [max(b["n"], 0.5) for b in cal], width=0.085, color=MUTED)
    ax2.set_yscale("log"); ax2.set_xlim(0, 1); ax2.set_ylabel("Number of lines\n(log scale)")
    ax2.set_xlabel("Predicted chance of denial (in bins of 10 points)")
    ax2.set_title("How many lines fall in each bin: 87% score below 30%", loc="left", fontsize=9)
    fig.text(0.01, 0.005, "Test set. Average gap between predicted and actual across bins: 0.7 percentage points (all types), "
             "1.0 carrier, 1.8 outpatient, 1.2 DME.", fontsize=8, color=MUTED)
    save(fig, "fig2_calibration.png")


# ---- Figure 3: cost versus a single cutoff, with cost sensitivity ----------------------
def net_at_cutoff(t, rho, r):
    flag = p > t
    L = loss_if_missed(A, rho)
    return (y[flag] * L[flag]).sum() - r * flag.sum()


def fig_cost_threshold():
    from matplotlib.lines import Line2D
    ts = np.linspace(0.0, 0.9, 181)
    ms = [(10, AQUA), (20, BLUE), (30, ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    sweep = {(round(s["rho"], 2), s["m"], round(s["b"], 2), round(s["c"], 3)): s for s in RES["scored"]["overall"]["sweep"]}
    for ax, rho, title in zip(axes, (0.24, 0.55), ("If 24% of overpayments are recovered", "If 55% of overpayments are recovered")):
        for m, col in ms:
            r = review_cost(m)
            L = loss_if_missed(A, rho)
            vals = np.array([net_at_cutoff(t, rho, r) for t in ts]) / 1e6
            ax.plot(ts, vals, color=col, lw=1.8)
            best = int(vals.argmax())
            ax.plot(ts[best], vals[best], "o", color=col, ms=5)
            flag = p * L > r
            rule = ((y * L)[flag].sum() - r * flag.sum()) / 1e6
            ax.axhline(rule, color=col, lw=1.4, ls=":")
            s = sweep[(rho, float(m), B, C)]
            check(f"rule net savings rho={rho} m={m}", rule * 1e6, s["net_savings"], 1e-6)
            if m == 20:
                review_all = ((y * L).sum() - r * len(p)) / 1e6
                check(f"review-all net rho={rho}", review_all * 1e6, s["review_all_net_savings"], 1e-6)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("Score cutoff t used for every line (review if score is above t)")
        ax.set_ylabel("Net savings vs reviewing nothing ($ millions)")
    handles = [Line2D([], [], color=c, lw=2, label=f"{m} minutes per review (${review_cost(m):.2f} each)") for m, c in ms]
    handles += [Line2D([], [], color=MUTED, lw=1.8, label="Solid: one cutoff t for every line"),
                Line2D([], [], color=MUTED, lw=1.4, ls=":", label="Dotted: judge each line by its own dollars (our rule)"),
                Line2D([], [], color=MUTED, marker="o", lw=0, label="Dot: best single cutoff, found with hindsight")]
    axes[0].legend(handles=handles, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=7.8)
    fig.suptitle("Judging each line by its own dollars beats any single score cutoff, at every review cost tested",
                 x=0.01, ha="left", fontsize=11.5, y=1.02)
    fig.text(0.01, -0.03, "Test set. Net savings = overpayments recovered by reviewing, minus the cost of the reviews. Our rule: review a line when "
             "chance of denial x loss if missed exceeds the review cost, so a \\$2,000 line is reviewed at a much lower score than a \\$20 line.",
             fontsize=8, color=MUTED, wrap=True)
    save(fig, "fig3_cost_vs_cutoff.png")


# ---- Figure 4: decision curve (appendix) -----------------------------------------------
def fig_decision_curve():
    panels = [("overall", "All claim types", INK), ("carrier", "Carrier (physician and supplier claims)", BLUE),
              ("outpatient", "Outpatient (hospital outpatient claims)", ORANGE), ("dme", "DME (equipment claims)", AQUA)]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), sharex=True)
    for ax, (key, title, col) in zip(axes.ravel(), panels):
        dc = RES["scored"][key]["decision_curve"]
        t = [d["t"] for d in dc]
        model = np.array([d["model"] for d in dc]) * 100
        ax.plot(t, model, color=col, lw=2.2, label="Review lines the model flags")
        ax.plot(t, np.array([d["review_all"] for d in dc]) * 100, color=MUTED, lw=1.4, ls="--", label="Review every line")
        ax.axhline(0, color=MUTED, lw=1)
        ax.text(0.59, 0, "review nothing = 0", ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_title(title, loc="left", fontsize=10)
        lo = min(model.min(), -1.0)
        ax.set_ylim(lo, model.max() * 1.1)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("How costly a false alarm is, as the score cutoff t")
    for ax in axes[:, 0]:
        ax.set_ylabel("Net denials found per 100 lines")
    fig.suptitle("Appendix: decision curve. The model finds more net denials than reviewing everything,\nbut in carrier and DME the gain is gone above a cutoff of about 0.3",
                 x=0.01, ha="left", fontsize=11, y=1.0)
    fig.text(0.01, -0.03, "Net denials found = denials caught per 100 lines, minus false alarms per 100 lines x t/(1-t). Higher t means false alarms are charged more.\n"
             "This counts denials, not dollars; dollars are in the cost figure.", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.9, bottom=0.1)
    save(fig, "fig4_decision_curve.png")


# ---- Figure 5: SHAP importance ---------------------------------------------------------
PLAIN = {
    "HCPCS_CD": "Procedure code (HCPCS)", "LINE_PLACE_OF_SRVC_CD": "Place of service",
    "CARR_CLM_BLG_NPI_NUM": "Billing provider ID (NPI)", "PRNCPAL_DGNS_CD": "Principal diagnosis code",
    "TAX_NUM": "Provider tax ID", "LINE_NUM": "Line number on the claim", "PRVDR_NUM": "Provider number",
    "ORG_NPI_NUM": "Organization ID (NPI)", "CARR_LINE_PRCNG_LCLTY_CD": "Pricing locality",
    "CARR_CLM_RFRNG_PIN_NUM": "Referring provider PIN", "ICD_DGNS_CD6": "Diagnosis code 6",
    "REV_CNTR_TOT_CHRG_AMT": "Total charges (outpatient)", "ICD_DGNS_CD5": "Diagnosis code 5",
    "ICD_DGNS_CD4": "Diagnosis code 4", "PRF_PHYSN_UPIN": "Performing physician ID (UPIN)",
}


def fig_shap():
    s = json.load(open(ROOT / "reports" / "shap_fit.json"))
    top = list(s["mean_abs_shap_overall_top20"].items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh([PLAIN.get(k, k) for k, _ in top], [v for _, v in top], color=BLUE, height=0.65)
    ax.grid(axis="y", visible=False)
    for i, (_, v) in enumerate(top):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Average push on a line's denial score (log-odds units; bigger = more influence)")
    ax.set_title("One field, the procedure code, drives the score; every other field is minor", loc="left", fontsize=10.5)
    fig.text(0.01, -0.04, "SHAP importance on a 20,000-line validation sample, model v2 (not recomputed on test). Names are plain-language "
             "versions of the data fields; definitions are in data/data_dictionary.md.", fontsize=8, color=MUTED, wrap=True)
    save(fig, "fig5_shap_importance.png")


# ---- Figure 6: does the explainer stay inside what its tools told it? -------------------
RUNS = [
    ("20260928-222319-final-v1", "Original tools\nand prompt", "9 of 364 untraceable:\nthe model's own\narithmetic"),
    ("20260928-222954-final-v1-newtools", "Tools return\nready-made numbers", "3 of 409 untraceable:\ncode meanings\nfrom memory"),
    ("20260928-223448-final-v2-newtools", "Prompt v2 forbids\narithmetic and\nguessed meanings", "0 of 540\nuntraceable"),
    ("20260928-224007-heldout-v2", "Same, on 30\nnew claims", "0 of 542\nuntraceable"),
    ("20260929-174122-heldout-v2-defs", "Plus sourced\ncode definitions", "0 of 511\nuntraceable"),
]


def fig_grounding():
    rows = []
    for d, label, note in RUNS:
        s = json.load(open(ROOT / "reports" / "phase4" / "runs" / d / "summary.json"))["pooled"]
        rows.append({"run": d, "label": label, "note": note, "total": s["tokens"], "grounded": s["grounded_tokens"],
                     "empty": s["empty_explanations"], "cov": s.get("tool_coverage")})
    df = pd.DataFrame(rows)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1.9, 1]})
    share = (df["grounded"] / df["total"] * 100).to_numpy()
    x = np.arange(len(df))
    ax.plot(x, share, color=BLUE, lw=2, marker="o", ms=7)
    for i, r_ in df.iterrows():
        ax.text(i, share[i] + 0.35, f"{share[i]:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([r_["label"] + "\n\n" + r_["note"] + (" +\n" + f"{r_['empty']} empty answers" if r_["empty"] else "") for _, r_ in df.iterrows()], fontsize=8)
    ax.set_ylim(96, 101); ax.set_xlim(-0.5, len(df) - 0.5); ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("% of checkable statements traceable to a tool output\n(axis starts at 96%)")
    ax.set_title("Each fix removed one kind of untraceable statement", loc="left", fontsize=10.5)
    # panel B: what the explainer is able to say
    a, b = df.iloc[3]["cov"], df.iloc[4]["cov"]
    groups = [("Top fields shown\nthat came with a meaning", a["features_with_meaning"] / a["features_shown"] * 100, b["features_with_meaning"] / b["features_shown"] * 100,
               f"{a['features_with_meaning']} of {a['features_shown']}", f"{b['features_with_meaning']} of {b['features_shown']}"),
              ("Code values shown\nthat came with a meaning", a["coded_values_with_meaning"] / a["coded_values_shown"] * 100, b["coded_values_with_meaning"] / b["coded_values_shown"] * 100,
               f"{a['coded_values_with_meaning']} of {a['coded_values_shown']}", f"{b['coded_values_with_meaning']} of {b['coded_values_shown']}")]
    xs = np.arange(len(groups))
    ax2.bar(xs - 0.2, [g[1] for g in groups], width=0.38, color="#c3c2b7", label="Before definitions were added")
    ax2.bar(xs + 0.2, [g[2] for g in groups], width=0.38, color=BLUE, label="After")
    for i, g in enumerate(groups):
        ax2.text(i - 0.2, g[1] + 2, f"{g[1]:.0f}%\n({g[3]})", ha="center", fontsize=8)
        ax2.text(i + 0.2, g[2] + 2, f"{g[2]:.0f}%\n({g[4]})", ha="center", fontsize=8)
    ax2.set_xticks(xs); ax2.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax2.set_ylim(0, 125); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.set_ylabel("%")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)
    ax2.set_title("Staying traceable is easy by saying little,\nso check how much it can say", loc="left", fontsize=10.5)
    fig.suptitle("The claim explainer now says only what its tools returned, and the tools return more of what a reviewer needs",
                 x=0.01, ha="left", fontsize=11.5, y=1.03)
    fig.text(0.01, -0.17, "30 claims per run, one answer each. Traceable means the number or code appears in a tool output; it does not mean correct. "
             "Policy over-statements were found by reading answers and are not counted here.", fontsize=8, color=MUTED, wrap=True)
    save(fig, "fig6_grounding_by_failure_type.png")
    return df


if __name__ == "__main__":
    fig_ranking(); fig_calibration(); fig_cost_threshold(); fig_decision_curve(); fig_shap(); df = fig_grounding()
    (OUT / "figure_checks.txt").write_text("\n".join(CHECKS) + "\n")
    print("\n".join(CHECKS)); print(df[["run", "total", "grounded", "empty"]].to_string())
