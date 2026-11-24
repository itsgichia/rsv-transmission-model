"""Parameter-recovery check for the Bayesian calibration.

Simulates monthly notifications from known parameters, fits the model to them
with sequential Monte Carlo, and reports whether the truth sits inside the
posterior. This is the test that must pass before the fit is pointed at real
surveillance data: if the machinery cannot recover parameters it generated, it
cannot be trusted to estimate ones it did not.

Usage::

    python scripts/fit_recovery.py --draws 200 --chains 2
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

from rsvmodel.fit import DEFAULT_BANDS, FitParams, fit, save_netcdf, simulate_observations  # noqa: E402

TRUTH = FitParams(r0=1.6, seasonal_amplitude=0.4, peak_day=100.0, immunity_days=230.0)
RHO = np.array([0.30, 0.15, 0.08, 0.03, 0.01, 0.005, 0.005, 0.01, 0.03])
ALPHA = 20.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--burn-in-years", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fit" / "recovery")
    args = parser.parse_args(argv)

    import arviz as az

    obs = simulate_observations(
        TRUTH, RHO, ALPHA, 2023, 1, args.months, 8_300_000, DEFAULT_BANDS, seed=args.seed, burn_in_years=args.burn_in_years
    )
    began = time.perf_counter()
    idata = fit(
        obs,
        draws=args.draws,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.seed,
        burn_in_years=args.burn_in_years,
        checkpoint=args.out / "recovery_posterior.nc",
    )
    elapsed = time.perf_counter() - began

    var_names = ["r0", "seasonal_amplitude", "peak_day", "immunity_days", "rho", "alpha"]
    summary = az.summary(idata, var_names=var_names, ci_prob=0.95, ci_kind="hdi", round_to="none")
    truth = {"r0": TRUTH.r0, "seasonal_amplitude": TRUTH.seasonal_amplitude, "peak_day": TRUTH.peak_day,
             "immunity_days": TRUTH.immunity_days, "alpha": ALPHA}
    truth.update({f"rho[{b.label}]": float(r) for b, r in zip(DEFAULT_BANDS, RHO)})
    summary["truth"] = [truth.get(i, np.nan) for i in summary.index]
    summary["covered"] = (summary["hdi95_lb"] <= summary["truth"]) & (summary["truth"] <= summary["hdi95_ub"])

    args.out.mkdir(parents=True, exist_ok=True)
    save_netcdf(idata, args.out / "recovery.nc")
    summary.to_csv(args.out / "recovery_summary.csv")
    (args.out / "recovery_meta.json").write_text(json.dumps({
        "draws": args.draws, "chains": args.chains, "months": args.months, "burn_in_years": args.burn_in_years,
        "seed": args.seed, "seconds": elapsed, "n_covered": int(summary["covered"].sum()), "n_params": int(len(summary)),
    }, indent=2) + "\n")
    print(summary.to_string())
    print(f"\n{summary['covered'].sum()}/{len(summary)} parameters covered by the 95% HDI; {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
