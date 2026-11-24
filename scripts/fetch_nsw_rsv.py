"""Fetch NSW RSV notifications by month and age band from NSW Health's
Infectious Diseases Data (IDD) service and write them in the long CSV format
that ``rsvmodel.fit.Observations.from_csv`` reads.

The IDD dashboard (https://www1.health.nsw.gov.au/IDD/#/RSV) is an AngularJS
app over a JSON API. Its reports return HTML tables; this script parses them.
Three reports are combined:

* ``periodandage``: monthly counts for under 1, 1 and 2 to 4 years, which
  match the model's infant classes;
* ``ageandgender``: counts by five-year age band for a date window, requested
  one month at a time to give monthly series for ages 5 and over;
* ``period``: monthly totals, used only as a cross-check.

Ages 80 and over are dropped because the transmission model stops at 80.
Every raw response is cached under ``data/nsw/raw`` so the fetch is
reproducible and can be re-run without hitting the service again.

Usage::

    python scripts/fetch_nsw_rsv.py --from 2022-09 --to 2026-10

RSV became notifiable in NSW on 1 September 2022. Counts exclude people whose
age or sex was unknown and non-residents, exclude the last 48 hours, and are
preliminary. Citation: NSW Health Notifiable Conditions Information Management
System (NCIMS), Communicable Diseases Branch and Centre for Epidemiology and
Evidence, NSW Health.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://idd-web.citc.health.nsw.gov.au/APi/Disease/GetReportContent"
ALL_LHDS = "X700,X710,X720,X730,X740,X750,X760,X770,X800,X810,X820,X830,X840,X850,X860,X910"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: Output bands: (label, lower_years, upper_years). Five-year surveillance bands
#: are summed into the model's coarser adult classes.
BANDS = [
    ("<1 year", 0.0, 1.0),
    ("1 year", 1.0, 2.0),
    ("2-4 years", 2.0, 5.0),
    ("5-9 years", 5.0, 10.0),
    ("10-19 years", 10.0, 20.0),
    ("20-39 years", 20.0, 40.0),
    ("40-59 years", 40.0, 60.0),
    ("60-79 years", 60.0, 80.0),
]
FIVE_YEAR_TO_BAND = {
    "05-09 yrs": "5-9 years",
    "10-14 yrs": "10-19 years",
    "15-19 yrs": "10-19 years",
    "20-24 yrs": "20-39 years",
    "25-29 yrs": "20-39 years",
    "30-34 yrs": "20-39 years",
    "35-39 yrs": "20-39 years",
    "40-44 yrs": "40-59 years",
    "45-49 yrs": "40-59 years",
    "50-54 yrs": "40-59 years",
    "55-59 yrs": "40-59 years",
    "60-64 yrs": "60-79 years",
    "65-69 yrs": "60-79 years",
    "70-74 yrs": "60-79 years",
    "75-79 yrs": "60-79 years",
}
INFANT_TO_BAND = {"< 1 year": "<1 year", "1 year": "1 year", "2 - 4 years": "2-4 years"}


class _Table(HTMLParser):
    """Collect the text of every cell of every table into rows."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_tables(html_tables: list[str]) -> list[list[list[str]]]:
    """Rows of cells, blanks kept as "" so column positions survive row-spanning labels."""
    out = []
    for html in html_tables:
        p = _Table()
        p.feed(html)
        out.append([["" if c in ("", "\xa0") else c for c in row] for row in p.rows])
    return out


def _year_rows(table: list[list[str]]):
    """Yield (label_before_year, year, month_values) for each data row of a by-period table."""
    header = table[0]
    year_col = header.index("Year")
    month_names = [h for h in header[year_col + 1 :] if h in MONTHS]
    for row in table[1:]:
        idx = next((i for i, c in enumerate(row) if re.fullmatch(r"\d{4}", c)), None)
        if idx is None:
            continue
        label = row[idx - 1] if idx >= 1 else ""
        values = row[idx + 1 : idx + 1 + len(month_names)]
        yield label, int(row[idx]), dict(zip(month_names, values))


def to_int(text: str) -> int:
    return int(text.replace(",", "").strip() or 0)


def month_iter(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's bundle where the interpreter has no system roots (python.org macOS builds)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(report_by: str, params: dict, cache_dir: Path, pause: float) -> dict:
    """GET one report, caching the JSON response by report type and parameters."""
    key = report_by + "_" + re.sub(r"[^0-9A-Za-z]+", "_", params["prFrom"] + "_" + params["prTo"])
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    query = urlencode({"reportBy": report_by, "reportParameter": json.dumps(params, separators=(",", ":"))})
    req = Request(f"{BASE}?{query}", headers={"Accept": "application/json", "User-Agent": "rsvmodel-fetch/1.0"})
    with urlopen(req, timeout=120, context=_ssl_context()) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    payload.pop("graphs", None)  # base64 PNGs, not needed
    payload["_fetched"] = date.today().isoformat()
    payload["_params"] = params
    path.write_text(json.dumps(payload))
    time.sleep(pause)
    return payload


def infant_counts(payload: dict) -> dict[tuple[str, str], int]:
    """(YYYY-MM, band) -> count from the periodandage report."""
    out: dict[tuple[str, str], int] = {}
    for table in parse_tables(payload["dataTables"]):
        current = None
        for label, year, values in _year_rows(table):
            if label:
                current = INFANT_TO_BAND.get(label)
            if current is None:
                continue
            for name, v in values.items():
                if v:
                    out[(f"{year:04d}-{MONTHS.index(name) + 1:02d}", current)] = to_int(v)
    return out


def adult_counts(payload: dict, month: str) -> dict[tuple[str, str], int]:
    """(YYYY-MM, band) -> count from one month's ageandgender report (Persons row)."""
    out: dict[tuple[str, str], int] = {}
    under5 = None
    for table in parse_tables(payload["dataTables"]):
        header = table[0]
        gcol = header.index("Gender")
        bands = header[gcol + 1 :]
        persons = next((r for r in table[1:] if "Persons" in r), None)
        if persons is None:
            raise ValueError(f"no Persons row for {month}")
        values = persons[persons.index("Persons") + 1 :]
        for h, v in zip(bands, values):
            if h == "00-04 yrs":
                under5 = to_int(v)
            elif h in FIVE_YEAR_TO_BAND:
                key = (month, FIVE_YEAR_TO_BAND[h])
                out[key] = out.get(key, 0) + to_int(v)
    out[(month, "_under5_check")] = under5 if under5 is not None else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="start", default="2022-09", help="first month, YYYY-MM")
    ap.add_argument("--to", dest="end", required=True, help="last complete month, YYYY-MM")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "nsw")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sy, sm = args.start.split("-")
    ey, em = args.end.split("-")
    common = {"prDisease": "RSV", "prStrain": "Not specified"}

    infants = fetch(
        "periodandage",
        {**common, "prAgeGroup": "< 1 year,1 year,2 - 4 years", "prReportPeriod": "3",
         "prFrom": f"{int(sm)}/{sy}", "prTo": f"{int(em)}/{ey}"},
        raw, args.pause,
    )
    totals = fetch(
        "period",
        {**common, "prLHD": ALL_LHDS, "prReportPeriod": "3", "prFrom": f"{int(sm)}/{sy}", "prTo": f"{int(em)}/{ey}"},
        raw, args.pause,
    )
    counts = infant_counts(infants)

    monthly_total: dict[str, int] = {}
    for table in parse_tables(totals["dataTables"]):
        for _label, year, values in _year_rows(table):
            for name, v in values.items():
                if v:
                    monthly_total[f"{year:04d}-{MONTHS.index(name) + 1:02d}"] = to_int(v)

    checks = []
    for y, m in month_iter(args.start, args.end):
        month = f"{y:04d}-{m:02d}"
        payload = fetch(
            "ageandgender",
            {**common, "prLHD": ALL_LHDS, "prGender": "Male,Female", "prFrom": f"{m}/{y}", "prTo": f"{m}/{y}"},
            raw, args.pause,
        )
        if payload.get("isEmptyDataset"):
            for label, *_ in BANDS[3:]:
                counts[(month, label)] = 0
            continue
        got = adult_counts(payload, month)
        under5 = got.pop((month, "_under5_check"))
        counts.update(got)
        infant_sum = sum(counts.get((month, b), 0) for b in ("<1 year", "1 year", "2-4 years"))
        checks.append((month, under5, infant_sum, monthly_total.get(month)))
        print(f"{month}: under-5 {under5:>6} (drill-down {infant_sum:>6})  total {monthly_total.get(month)}", file=sys.stderr)

    out_csv = args.out / "rsv_monthly_by_age.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["month", "age_band", "lower_years", "upper_years", "count"])
        for y, m in month_iter(args.start, args.end):
            month = f"{y:04d}-{m:02d}"
            for label, lo, hi in BANDS:
                w.writerow([month, label, lo, hi, counts.get((month, label), 0)])

    with (args.out / "rsv_monthly_total.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["month", "count"])
        for month in sorted(monthly_total):
            w.writerow([month, monthly_total[month]])

    mismatches = [c for c in checks if c[1] != c[2]]
    meta = {
        "source": "NSW Health Infectious Diseases Data, https://www1.health.nsw.gov.au/IDD/#/RSV",
        "api": BASE,
        "fetched": date.today().isoformat(),
        "months": [args.start, args.end],
        "bands": [b[0] for b in BANDS],
        "dropped": "ages 80 and over (model age structure ends at 80)",
        "caveats": re.sub(r"<[^>]+>", "", infants["labels"]["footer"]),
        "citation": infants["labels"]["citations"],
        "notifiable_from": infants["labels"]["notifiableFrom"],
        "under5_mismatch_months": mismatches,
    }
    (args.out / "rsv_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_csv}; {len(checks)} months; under-5 cross-check mismatches: {len(mismatches)}", file=sys.stderr)


if __name__ == "__main__":
    main()
