"""Fit the model to NSW RSV notifications by month and age band.

Reads ``data/nsw/rsv_monthly_by_age.csv`` (written by ``fetch_nsw_rsv.py``),
keeps the months in the chosen window, samples the posterior with sequential
Monte Carlo and writes a summary table, the posterior as NetCDF and a posterior
predictive check figure. Run ``fit_recovery.py`` first: this script is only
worth running once the machinery recovers parameters it generated itself.

The default window starts in January 2023. The 2021 and 2022 seasons followed
the lifting of COVID-19 restrictions and are out of phase with the endemic
seasonality the model assumes.

Usage::

    python scripts/fit_nsw.py --draws 1000 --chains 2 --cores 2
    python scripts/fit_nsw.py --draws 20 --end 2023-12 --burn-in-years 6   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsvmodel.fit import Observations, fit, save_netcdf  # noqa: E402

#: NSW estimated resident population, mid-2024 (ABS 3101.0), rounded.
NSW_POPULATION = 8_500_000
SOURCE = "NSW Health Infectious Diseases Data dashboard, RSV notifications by month of onset and age"
VAR_NAMES = ["r0", "seasonal_amplitude", "peak_day", "immunity_days", "rho", "alpha"]


def ppc_figure(idata, obs: Observations, path: Path) -> None:
    """Observed counts against the posterior predictive median and 95% interval, one panel per band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = idata["posterior_predictive"]["counts"].values  # (chain, draw, band, month)
    pred = pred.reshape(-1, *pred.shape[-2:])
    lo, mid, hi = np.percentile(pred, [2.5, 50, 97.5], axis=0)
    labels = obs.month_labels()
    x = np.arange(obs.n_months)
    ticks = [i for i, m in enumerate(labels) if m.endswith("-01") or i == 0]

    n = len(obs.bands)
    ncols = 2 if n > 4 else 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.3 * nrows), sharex=True, squeeze=False)
    for ax, band, k in zip(axes.ravel(), obs.bands, range(n)):
        ax.fill_between(x, lo[k], hi[k], color="#5b8def", alpha=0.25, linewidth=0, label="95% posterior predictive")
        ax.plot(x, mid[k], color="#5b8def", linewidth=1.5, label="Posterior predictive median")
        ax.plot(x, obs.counts[k], "o", color="#222222", markersize=3, label="Observed")
        ax.set_title(band.label, fontsize=10, loc="left")
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i][:4] for i in ticks])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("NSW RSV notifications per month and the fitted model", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "nsw" / "rsv_monthly_by_age.csv")
    parser.add_argument("--population", type=float, default=NSW_POPULATION)
    parser.add_argument("--start", default="2023-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default=None, help="last month, YYYY-MM (default: last in the file)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--burn-in-years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fit" / "nsw")
    parser.add_argument("--figure", type=Path, default=REPO_ROOT / "figures" / "nsw_fit_ppc.png")
    args = parser.parse_args(argv)

    import arviz as az

    obs = Observations.from_csv(args.data, args.population, SOURCE).window(args.start, args.end)
    print(f"{obs.n_months} months, {len(obs.bands)} bands, {int(obs.counts.sum()):,} notifications, "
          f"{obs.month_labels()[0]} to {obs.month_labels()[-1]}")

    began = time.perf_counter()
    idata = fit(
        obs,
        draws=args.draws,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.seed,
        burn_in_years=args.burn_in_years,
        checkpoint=args.out / "nsw_posterior.nc",
    )
    elapsed = time.perf_counter() - began

    summary = az.summary(idata, var_names=VAR_NAMES, ci_prob=0.95, ci_kind="hdi", round_to="none")
    args.out.mkdir(parents=True, exist_ok=True)
    save_netcdf(idata, args.out / "nsw.nc")
    summary.to_csv(args.out / "nsw_summary.csv")
    ppc_figure(idata, obs, args.figure)

    pred = idata["posterior_predictive"]["counts"].values.reshape(-1, len(obs.bands), obs.n_months)
    lo, hi = np.percentile(pred, [2.5, 97.5], axis=0)
    inside = float(((lo <= obs.counts) & (obs.counts <= hi)).mean())
    (args.out / "nsw_meta.json").write_text(json.dumps({
        "data": str(args.data.relative_to(REPO_ROOT)) if args.data.is_relative_to(REPO_ROOT) else str(args.data),
        "population": args.population, "start": obs.month_labels()[0], "end": obs.month_labels()[-1],
        "n_months": obs.n_months, "n_bands": len(obs.bands), "draws": args.draws, "chains": args.chains,
        "burn_in_years": args.burn_in_years, "seed": args.seed, "seconds": elapsed,
        "ppc_coverage_95": inside,
    }, indent=2) + "\n")
    print(summary.to_string())
    print(f"\n{inside:.0%} of observed counts inside the 95% posterior predictive interval; {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
