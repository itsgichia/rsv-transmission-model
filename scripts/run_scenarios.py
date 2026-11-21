"""Run the default immunisation scenarios and save figures and a results table.

Usage: python scripts/run_scenarios.py
"""

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsvmodel import Params, maternal_vaccine, nirsevimab, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
BASE, SCEN = "#9ca3af", "#2563eb"

SCENARIOS = {
    "Maternal vaccine (60% coverage)": dict(maternal=maternal_vaccine(0.6)),
    "Nirsevimab at birth (60% coverage)": dict(monoclonal=nirsevimab(0.6)),
    "Nirsevimab, April to September births only": dict(monoclonal=nirsevimab(0.6, seasonal_only=True)),
    "Maternal vaccine 60% + nirsevimab for the rest (50%)": dict(
        maternal=maternal_vaccine(0.6), monoclonal=nirsevimab(0.5)
    ),
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    params = Params()
    rows = []
    results = {}
    for name, kw in SCENARIOS.items():
        r = simulate(params, years=5, **kw)
        results[name] = r
        for s in r["summary"]:
            rows.append({"scenario": name, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}})

    with open(RES / "scenario_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Figure 1: weekly hospitalisations in 0-5 month olds, baseline vs nirsevimab
    r = results["Nirsevimab at birth (60% coverage)"]
    weeks = np.array(r["weeks"]) / 52
    base = np.add(r["hospitalisations"]["baseline"]["0-2 months"], r["hospitalisations"]["baseline"]["3-5 months"])
    scen = np.add(r["hospitalisations"]["scenario"]["0-2 months"], r["hospitalisations"]["scenario"]["3-5 months"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(weeks, base, color=BASE, lw=1.8, label="No immunisation")
    ax.plot(weeks, scen, color=SCEN, lw=2, label="Nirsevimab at birth, 60% coverage")
    ax.set_xlabel("Years after programme start", color=MUTED)
    ax.set_ylabel("Weekly hospitalisations\n(0-5 months, per million)", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Seasonal RSV hospitalisations in infants under 6 months", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "weekly_hospitalisations.png", dpi=160)
    plt.close(fig)

    # Figure 2: percent averted by age group per scenario
    groups = ["0-2 months", "3-5 months", "6-11 months", "12-23 months"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(groups))
    width = 0.2
    palette = ["#2563eb", "#0d9488", "#d97706", "#7c3aed"]
    for i, (name, r) in enumerate(results.items()):
        vals = [next(s["percent_averted"] for s in r["summary"] if s["group"] == g) for g in groups]
        ax.bar(x + (i - 1.5) * width, vals, width, color=palette[i], label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Hospitalisations averted (%)", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Who benefits: percentage of hospitalisations averted by age", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "percent_averted_by_age.png", dpi=160)
    plt.close(fig)

    # Figure 3: annual vs biennial dynamics as R0 changes
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
    for ax, r0 in zip(axes, (1.5, 1.8)):
        rr = simulate(Params(r0=r0), years=4)
        w = np.array(rr["weeks"]) / 52
        ax.plot(w, rr["hospitalisations"]["baseline"]["0-2 months"], color=SCEN, lw=1.8)
        ax.set_title(f"R0 = {r0}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Years", color=MUTED)
        style(ax)
    axes[0].set_ylabel("Weekly hospitalisations\n(0-2 months)", color=MUTED)
    fig.suptitle("Biennial and annual epidemics from the same model", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "annual_vs_biennial.png", dpi=160)
    plt.close(fig)

    # Figure 4: homogeneous vs assortative mixing, same programme
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for mixing, colour in (("homogeneous", BASE), ("assortative", SCEN)):
        rr = simulate(Params(mixing=mixing), monoclonal=nirsevimab(0.6), years=5)
        w = np.array(rr["weeks"]) / 52
        hb = rr["hospitalisations"]["baseline"]
        under2 = np.sum([hb[g] for g in hb], axis=0)
        ax.plot(w, under2, color=colour, lw=1.8, label=f"{mixing} mixing")
        pct = next(s["percent_averted"] for s in rr["summary"] if s["group"] == "Under 2 years")
        rows.append({"scenario": f"Nirsevimab 60%, {mixing} mixing", "group": "Under 2 years",
                     "baseline_per_year": "", "scenario_per_year": "", "averted_per_year": "",
                     "percent_averted": round(pct, 1)})
    ax.set_xlabel("Years after programme start", color=MUTED)
    ax.set_ylabel("Weekly hospitalisations\n(under 2 years, per million)", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Same R0, different mixing: infants see less infection when contacts are assortative", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "mixing_comparison.png", dpi=160)
    plt.close(fig)

    with open(RES / "scenario_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
