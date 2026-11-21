"""Command line interface: run one immunisation scenario and print the summary.

Examples
--------
    rsvmodel run --maternal 0.6
    rsvmodel run --nirsevimab 0.6 --seasonal-only --years 3 --format csv
    rsvmodel run --maternal 0.6 --nirsevimab 0.5 --mixing assortative --format json
    rsvmodel defaults
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict

from .model import MIXING_MODES, Params, maternal_vaccine, nirsevimab, simulate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsvmodel", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate baseline and an immunisation programme from the same endemic state")
    run.add_argument("--r0", type=float, default=Params.r0, help="basic reproduction number (default %(default)s)")
    run.add_argument("--amplitude", type=float, default=Params.seasonal_amplitude, help="seasonal forcing amplitude b1")
    run.add_argument("--immunity-days", type=float, default=Params.immunity_days, help="mean duration of post-infection immunity")
    run.add_argument("--mixing", choices=MIXING_MODES, default="homogeneous")
    run.add_argument("--population", type=float, default=Params.population)
    run.add_argument("--maternal", type=float, default=0.0, metavar="COVERAGE", help="maternal vaccine coverage, 0-1")
    run.add_argument("--maternal-efficacy", type=float, default=0.8)
    run.add_argument("--nirsevimab", type=float, default=0.0, metavar="COVERAGE", help="nirsevimab coverage at birth, 0-1")
    run.add_argument("--nirsevimab-efficacy", type=float, default=0.8)
    run.add_argument("--seasonal-only", action="store_true", help="give nirsevimab only to infants born April to September")
    run.add_argument("--years", type=int, default=5)
    run.add_argument("--format", choices=("table", "csv", "json"), default="table")

    sub.add_parser("defaults", help="print the default parameters as JSON")
    return parser


def _run(args: argparse.Namespace) -> int:
    params = Params(
        r0=args.r0,
        seasonal_amplitude=args.amplitude,
        immunity_days=args.immunity_days,
        mixing=args.mixing,
        population=args.population,
    )
    maternal = maternal_vaccine(args.maternal, args.maternal_efficacy) if args.maternal > 0 else None
    mono = nirsevimab(args.nirsevimab, args.nirsevimab_efficacy, args.seasonal_only) if args.nirsevimab > 0 else None
    result = simulate(params, maternal=maternal, monoclonal=mono, years=args.years)

    if args.format == "json":
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    rows = result["summary"]
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return 0

    print(f"R0 {params.r0}, mixing {params.mixing}, {args.years} years, population {params.population:,.0f}")
    print(f"{'Age group':<14}{'Baseline/yr':>13}{'Scenario/yr':>13}{'Averted/yr':>12}{'Averted %':>11}")
    for r in rows:
        print(
            f"{r['group']:<14}{r['baseline_per_year']:>13.1f}{r['scenario_per_year']:>13.1f}"
            f"{r['averted_per_year']:>12.1f}{r['percent_averted']:>10.1f}%"
        )
    doses = result["doses"]
    print(f"\nDoses per year: maternal {doses['maternal']:,.0f}, nirsevimab {doses['nirsevimab']:,.0f}")
    if doses["per_admission_averted"] is not None:
        print(f"Doses per admission averted: {doses['per_admission_averted']:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "defaults":
        json.dump(asdict(Params()), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
