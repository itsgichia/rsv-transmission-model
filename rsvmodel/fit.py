"""Bayesian calibration of the RSV model to monthly notification counts.

The transmission model in :mod:`rsvmodel.model` is deterministic. This module
wraps it in an observation model and fits the parameters that surveillance data
can inform:

* the basic reproduction number ``r0``,
* the seasonal forcing amplitude ``seasonal_amplitude`` and its ``peak_day``,
* the duration of post-infection immunity ``immunity_days``,
* one reporting fraction per age band (the share of infections that become a
  laboratory-confirmed notification), and
* a negative-binomial dispersion for month-to-month noise.

The latent and infectious periods are held at the literature values used by the
model; a few years of monthly counts cannot separate them from ``r0``.

Likelihood
----------
For age band ``a`` and month ``m``, with ``I[a, m]`` the model's new infections
in that band over that month,

    counts[a, m] ~ NegativeBinomial(mu = rho[a] * I[a, m], alpha)

Inference
---------
Every likelihood evaluation runs the ODE from an endemic equilibrium (a burn-in
of several decades, cached per parameter set) and then forward over the fitted
months. That costs a few tenths of a second and offers no gradient, so the
posterior is sampled with PyMC's sequential Monte Carlo sampler, which needs
only the log density. ArviZ carries the diagnostics.

The model year is 365.25 days with no calendar. Months are treated as windows
of 365.25 / 12 days starting at the first of January of the model year, which
misplaces a month boundary by at most a day or two. Notifications are by month
of disease onset, so there is no reporting-delay adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit, gammaln

from .model import (
    AGE_LOWER_YEARS,
    AGE_WIDTHS_DAYS,
    DAYS_PER_MONTH,
    DAYS_PER_YEAR,
    N_AGE,
    Params,
    _E,
    _burn_in,
    _params_key,
    _solve,
)

__all__ = [
    "AgeBand",
    "DEFAULT_BANDS",
    "Observations",
    "FitParams",
    "Priors",
    "band_matrix",
    "monthly_incidence",
    "expected_counts",
    "negative_binomial_logpmf",
    "log_likelihood",
    "simulate_observations",
    "build_model",
    "fit",
    "posterior_params",
]


# ---------------------------------------------------------------------------
# Age bands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgeBand:
    """A reporting age band, in years, lower bound inclusive and upper exclusive."""

    label: str
    lower_years: float
    upper_years: float

    def __post_init__(self) -> None:
        if not 0 <= self.lower_years < self.upper_years:
            raise ValueError(f"bad band {self.label!r}: [{self.lower_years}, {self.upper_years})")


#: Bands aligned with the model's own age classes. Surveillance bands that are
#: coarser (for example 0 to 4 years) are sums of these; bands that are finer
#: than a model class cannot be represented and must be merged upstream.
DEFAULT_BANDS: tuple[AgeBand, ...] = (
    AgeBand("0-5 months", 0.0, 0.5),
    AgeBand("6-11 months", 0.5, 1.0),
    AgeBand("12-23 months", 1.0, 2.0),
    AgeBand("2-4 years", 2.0, 5.0),
    AgeBand("5-9 years", 5.0, 10.0),
    AgeBand("10-19 years", 10.0, 20.0),
    AgeBand("20-39 years", 20.0, 40.0),
    AgeBand("40-59 years", 40.0, 60.0),
    AgeBand("60-79 years", 60.0, 80.0),
)


def band_matrix(bands: Sequence[AgeBand]) -> np.ndarray:
    """Return the ``(n_bands, N_AGE)`` matrix mapping model classes to bands.

    Entry ``[a, i]`` is the fraction of model class ``i`` that falls inside band
    ``a`` by age span, so a class is split proportionally when a band boundary
    cuts through it. Rows need not sum to anything; columns sum to at most one.
    """
    lower = AGE_LOWER_YEARS
    upper = lower + AGE_WIDTHS_DAYS / DAYS_PER_YEAR
    matrix = np.zeros((len(bands), N_AGE))
    for a, band in enumerate(bands):
        overlap = np.clip(np.minimum(upper, band.upper_years) - np.maximum(lower, band.lower_years), 0, None)
        matrix[a] = overlap / (upper - lower)
    return matrix


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Observations:
    """Monthly notification counts by age band.

    Attributes:
        start_year: Calendar year of the first month.
        start_month: Calendar month (1 to 12) of the first month.
        bands: Age bands, one per row of ``counts``.
        counts: Integer array of shape ``(n_bands, n_months)``.
        population: Population of the jurisdiction the counts cover.
        source: Free-text provenance, written into the results.
    """

    start_year: int
    start_month: int
    bands: tuple[AgeBand, ...]
    counts: np.ndarray
    population: float
    source: str = ""

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        if counts.ndim != 2 or counts.shape[0] != len(self.bands):
            raise ValueError("counts must have one row per band")
        if not 1 <= self.start_month <= 12:
            raise ValueError("start_month must be in 1..12")
        if (counts < 0).any():
            raise ValueError("counts must be non-negative")
        object.__setattr__(self, "counts", counts.astype(float))

    @property
    def n_months(self) -> int:
        return int(self.counts.shape[1])

    def month_labels(self) -> list[str]:
        """``YYYY-MM`` labels for each column."""
        labels = []
        year, month = self.start_year, self.start_month
        for _ in range(self.n_months):
            labels.append(f"{year:04d}-{month:02d}")
            month += 1
            if month > 12:
                month, year = 1, year + 1
        return labels

    @classmethod
    def from_csv(cls, path: Path | str, population: float, source: str = "") -> "Observations":
        """Read a long-format CSV with columns ``month`` (YYYY-MM), ``age_band``,
        ``lower_years``, ``upper_years`` and ``count``."""
        import csv

        rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"{path} is empty")
        months = sorted({r["month"] for r in rows})
        band_specs: dict[str, AgeBand] = {}
        for r in rows:
            band_specs.setdefault(
                r["age_band"], AgeBand(r["age_band"], float(r["lower_years"]), float(r["upper_years"]))
            )
        bands = tuple(sorted(band_specs.values(), key=lambda b: b.lower_years))
        index = {b.label: i for i, b in enumerate(bands)}
        counts = np.zeros((len(bands), len(months)))
        col = {m: j for j, m in enumerate(months)}
        for r in rows:
            counts[index[r["age_band"]], col[r["month"]]] += float(r["count"])
        year, month = (int(x) for x in months[0].split("-"))
        expected = cls(year, month, bands, counts, population, source).month_labels()
        if expected != months:
            raise ValueError("months are not consecutive")
        return cls(year, month, bands, counts, population, source)


# ---------------------------------------------------------------------------
# Parameters and priors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitParams:
    """The transmission parameters being fitted.

    Everything else in :class:`rsvmodel.model.Params` is held at ``base``.
    """

    r0: float
    seasonal_amplitude: float
    peak_day: float
    immunity_days: float

    NAMES = ("r0", "seasonal_amplitude", "peak_day", "immunity_days")

    def to_params(self, base: Params) -> Params:
        return replace(
            base,
            r0=self.r0,
            seasonal_amplitude=self.seasonal_amplitude,
            peak_day=self.peak_day,
            immunity_days=self.immunity_days,
        )

    @classmethod
    def from_array(cls, theta: Sequence[float]) -> "FitParams":
        return cls(*(float(x) for x in theta))

    def to_array(self) -> np.ndarray:
        return np.array([self.r0, self.seasonal_amplitude, self.peak_day, self.immunity_days])


@dataclass(frozen=True)
class Priors:
    """Prior hyperparameters, written down before fitting.

    ``r0`` is truncated normal around the model's working value. The amplitude
    lives in (0, 1) with a Beta prior centred near the Western Australian fit.
    ``peak_day`` is normal around early April, which produces a June/July peak
    in infections. Immunity is log-normal around 230 days. Reporting fractions
    are logit-normal, wide, and centred on about 5%: most infections are mild
    and never tested. The negative-binomial ``alpha`` is a gamma with mean 20,
    which allows substantial over-dispersion without forcing it.
    """

    r0_mu: float = 1.6
    r0_sigma: float = 0.3
    r0_lower: float = 1.05
    r0_upper: float = 3.0
    amplitude_alpha: float = 2.0
    amplitude_beta: float = 3.0
    peak_day_mu: float = 100.0
    peak_day_sigma: float = 45.0
    peak_day_lower: float = 0.0
    peak_day_upper: float = 365.0
    log_immunity_mu: float = float(np.log(230.0))
    log_immunity_sigma: float = 0.3
    logit_rho_mu: float = -3.0
    logit_rho_sigma: float = 1.5
    alpha_shape: float = 2.0
    alpha_rate: float = 0.1


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def monthly_incidence(
    theta: FitParams,
    obs: Observations,
    base: Params | None = None,
    burn_in_years: int = 25,
) -> np.ndarray:
    """Return modelled new infections per band and month, shape ``(n_bands, n_months)``.

    New infections are taken as the flow out of the exposed compartment,
    ``E / latent_days``, integrated day by day over each month window. The run
    starts from the endemic state reached after ``burn_in_years`` and continues
    from the first of January to the end of the last observed month.
    """
    p = theta.to_params(base or Params(population=obs.population))
    if p.population != obs.population:
        p = replace(p, population=obs.population)
    y_start = _burn_in(_params_key(p), burn_in_years)
    t0 = burn_in_years * DAYS_PER_YEAR
    offset = (obs.start_month - 1) * DAYS_PER_MONTH
    t_end = t0 + offset + obs.n_months * DAYS_PER_MONTH
    t, y = _solve(p, y_start, t0, t_end, step=1.0)
    daily = y[_E] / p.latent_days  # (N_AGE, n_days) new infectious cases per day
    edges = t0 + offset + DAYS_PER_MONTH * np.arange(obs.n_months + 1)
    month_index = np.clip(np.searchsorted(edges, t, side="right") - 1, -1, obs.n_months)
    by_class = np.zeros((N_AGE, obs.n_months))
    for m in range(obs.n_months):
        mask = month_index == m
        by_class[:, m] = daily[:, mask].sum(axis=1)
    return band_matrix(obs.bands) @ by_class


def expected_counts(incidence: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Expected notifications: reporting fraction per band times incidence."""
    return np.asarray(rho)[:, None] * incidence


def negative_binomial_logpmf(counts: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Element-wise NB log-pmf in PyMC's ``(mu, alpha)`` parameterisation."""
    mu = np.clip(mu, 1e-9, None)
    k = np.asarray(counts)
    return (
        gammaln(k + alpha)
        - gammaln(alpha)
        - gammaln(k + 1)
        + alpha * np.log(alpha / (alpha + mu))
        + k * np.log(mu / (alpha + mu))
    )


def log_likelihood(
    theta: FitParams,
    rho: np.ndarray,
    alpha: float,
    obs: Observations,
    base: Params | None = None,
    burn_in_years: int = 25,
) -> float:
    """Total NB log-likelihood of the observations under one parameter set."""
    mu = expected_counts(monthly_incidence(theta, obs, base, burn_in_years), rho)
    return float(negative_binomial_logpmf(obs.counts, mu, alpha).sum())


def simulate_observations(
    theta: FitParams,
    rho: np.ndarray,
    alpha: float,
    start_year: int,
    start_month: int,
    n_months: int,
    population: float,
    bands: Sequence[AgeBand] = DEFAULT_BANDS,
    seed: int = 0,
    burn_in_years: int = 25,
) -> Observations:
    """Draw a synthetic dataset from the model, for parameter-recovery checks."""
    template = Observations(start_year, start_month, tuple(bands), np.zeros((len(bands), n_months)), population, "synthetic")
    mu = expected_counts(monthly_incidence(theta, template, None, burn_in_years), rho)
    rng = np.random.default_rng(seed)
    # NB(mu, alpha) as Gamma-Poisson: rate ~ Gamma(alpha, mu/alpha)
    rates = rng.gamma(alpha, mu / alpha)
    counts = rng.poisson(rates)
    return replace(template, counts=counts, source=f"synthetic, seed {seed}")


# ---------------------------------------------------------------------------
# PyMC model
# ---------------------------------------------------------------------------
def build_model(
    obs: Observations,
    priors: Priors | None = None,
    base: Params | None = None,
    burn_in_years: int = 25,
) -> Any:
    """Build the PyMC model. The ODE enters as a black-box pytensor Op."""
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph import Apply, Op

    priors = priors or Priors()
    n_bands = len(obs.bands)

    class IncidenceOp(Op):
        """theta (4,) -> modelled infections (n_bands, n_months)."""

        def make_node(self, theta):
            theta = pt.as_tensor_variable(theta)
            return Apply(self, [theta], [pt.dmatrix()])

        def perform(self, node, inputs, outputs):
            theta = FitParams.from_array(inputs[0])
            outputs[0][0] = monthly_incidence(theta, obs, base, burn_in_years)

    incidence_op = IncidenceOp()

    coords = {"band": [b.label for b in obs.bands], "month": obs.month_labels()}
    with pm.Model(coords=coords) as model:
        r0 = pm.TruncatedNormal("r0", mu=priors.r0_mu, sigma=priors.r0_sigma, lower=priors.r0_lower, upper=priors.r0_upper)
        amplitude = pm.Beta("seasonal_amplitude", alpha=priors.amplitude_alpha, beta=priors.amplitude_beta)
        peak_day = pm.TruncatedNormal(
            "peak_day", mu=priors.peak_day_mu, sigma=priors.peak_day_sigma, lower=priors.peak_day_lower, upper=priors.peak_day_upper
        )
        immunity_days = pm.LogNormal("immunity_days", mu=priors.log_immunity_mu, sigma=priors.log_immunity_sigma)
        logit_rho = pm.Normal("logit_rho", mu=priors.logit_rho_mu, sigma=priors.logit_rho_sigma, dims="band")
        rho = pm.Deterministic("rho", pm.math.sigmoid(logit_rho), dims="band")
        alpha = pm.Gamma("alpha", alpha=priors.alpha_shape, beta=priors.alpha_rate)

        theta = pt.stack([r0, amplitude, peak_day, immunity_days])
        incidence = pm.Deterministic("incidence", incidence_op(theta), dims=("band", "month"))
        mu = pm.Deterministic("mu", rho[:, None] * incidence, dims=("band", "month"))
        pm.NegativeBinomial("counts", mu=mu, alpha=alpha, observed=obs.counts, dims=("band", "month"))
    return model


def merge_groups(tree: Any, other: Any) -> Any:
    """Return ``tree`` with every group of ``other`` it does not already hold.

    ArviZ 1.x stores inference results as an ``xarray.DataTree`` rather than an
    ``InferenceData``, and ``DataTree`` has no ``extend``; this is the one-line
    replacement used to attach prior and posterior predictive groups.
    """
    import xarray as xr

    new = {name: xr.DataTree(child.to_dataset()) for name, child in other.children.items() if name not in tree.children}
    return tree.assign(new) if new else tree


def save_netcdf(tree: Any, path: Path | str) -> list[str]:
    """Write a ``DataTree`` as NetCDF, dropping variables NetCDF cannot encode.

    SMC sample statistics include ragged per-stage arrays (object dtype) that
    no NetCDF backend accepts. Object arrays made only of strings, such as the
    band labels, are kept. Dropped variable names are returned and recorded
    in the root attribute ``dropped_for_netcdf``.
    """
    import xarray as xr

    dropped: list[str] = []
    groups: dict[str, Any] = {}
    for name, child in tree.children.items():
        ds = child.to_dataset()
        bad = [
            v for v in ds.variables
            if ds[v].dtype == object and not all(isinstance(x, str) for x in np.asarray(ds[v].values).ravel())
        ]
        dropped.extend(f"{name}/{v}" for v in bad)
        groups[name] = ds.drop_vars(bad)
    out = xr.DataTree.from_dict(groups)
    out.attrs.update(tree.attrs)
    if dropped:
        out.attrs["dropped_for_netcdf"] = ",".join(dropped)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(str(path))
    return dropped


def fit(
    obs: Observations,
    draws: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    random_seed: int = 20260321,
    priors: Priors | None = None,
    base: Params | None = None,
    burn_in_years: int = 25,
    progressbar: bool = True,
    checkpoint: Path | None = None,
) -> Any:
    """Sample the posterior with sequential Monte Carlo and return a ``DataTree``.

    SMC needs only the log density, so the ODE Op requires no gradient. Each
    chain is an independent particle population, which gives an honest R-hat.
    When ``checkpoint`` is given the posterior is written there as NetCDF the
    moment sampling ends, before the predictive draws, so a failure in the
    cheap final steps cannot discard hours of sampling.
    """
    import pymc as pm

    model = build_model(obs, priors, base, burn_in_years)
    with model:
        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            cores=cores or chains,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )
        idata.attrs["observations_source"] = obs.source
        idata.attrs["population"] = float(obs.population)
        idata.attrs["burn_in_years"] = int(burn_in_years)
        if checkpoint is not None:
            save_netcdf(idata, checkpoint)
        idata = merge_groups(idata, pm.sample_prior_predictive(draws=min(draws, 500), random_seed=random_seed))
        idata = merge_groups(
            idata, pm.sample_posterior_predictive(idata, random_seed=random_seed, progressbar=progressbar)
        )
    return idata


def posterior_params(idata: Any, n: int = 200, seed: int = 0) -> list[tuple[FitParams, np.ndarray]]:
    """Thin ``n`` posterior draws to ``(FitParams, rho)`` pairs for scenario runs."""
    post = idata["posterior"].to_dataset().stack(sample=("chain", "draw"))
    total = post.sizes["sample"]
    rng = np.random.default_rng(seed)
    picks = rng.choice(total, size=min(n, total), replace=False)
    out = []
    for k in picks:
        theta = FitParams(
            r0=float(post["r0"][k]),
            seasonal_amplitude=float(post["seasonal_amplitude"][k]),
            peak_day=float(post["peak_day"][k]),
            immunity_days=float(post["immunity_days"][k]),
        )
        out.append((theta, np.asarray(post["rho"][:, k])))
    return out
