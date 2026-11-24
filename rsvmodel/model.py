"""Age-structured SEIRS model of RSV transmission with infant immunisation.

Structure follows the published RSV models for Western Australia by Hogan,
Glass, Moore and colleagues (SEIRS form, monthly age classes in infancy,
cosine seasonal forcing, age-specific susceptibility and infectiousness,
maternal vaccination as a protected compartment). Mixing is either
homogeneous or age-assortative, using a synthetic contact kernel with a
within-age-band term and a parent-child term; it is not fitted to survey
data such as POLYMOD, so it shows the direction of the effect rather than
a calibrated magnitude.

Time is in days. All rates are per day.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

try:  # the compiled right-hand side is optional; the numpy one is the reference
    import numba
except ImportError:  # pragma: no cover
    numba = None

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12

# ---------------------------------------------------------------------------
# Age structure: 24 monthly classes (0-23 months), then 2-4, 5-9, 10-19,
# 20-39, 40-59 and 60-79 years.
# ---------------------------------------------------------------------------
N_MONTHLY = 24
_OLDER_EDGES_YEARS = [2, 5, 10, 20, 40, 60, 80]
AGE_WIDTHS_DAYS = np.array(
    [DAYS_PER_MONTH] * N_MONTHLY
    + [(b - a) * DAYS_PER_YEAR for a, b in zip(_OLDER_EDGES_YEARS[:-1], _OLDER_EDGES_YEARS[1:])]
)
N_AGE = len(AGE_WIDTHS_DAYS)
AGE_LABELS = [f"{m}m" for m in range(N_MONTHLY)] + [
    f"{a}-{b - 1}y" for a, b in zip(_OLDER_EDGES_YEARS[:-1], _OLDER_EDGES_YEARS[1:])
]
LIFESPAN_DAYS = AGE_WIDTHS_DAYS.sum()
AGE_LOWER_YEARS = np.concatenate([[0.0], np.cumsum(AGE_WIDTHS_DAYS)[:-1]]) / DAYS_PER_YEAR

# Reporting groups for hospitalisation, by monthly class index.
HOSP_GROUPS = {
    "0-2 months": range(0, 3),
    "3-5 months": range(3, 6),
    "6-11 months": range(6, 12),
    "12-23 months": range(12, 24),
}

PROTECTION_MONTHS = 6  # protected compartments exist for the first 6 monthly classes
AGE_MID_YEARS = AGE_LOWER_YEARS + AGE_WIDTHS_DAYS / DAYS_PER_YEAR / 2
MIXING_MODES = ("homogeneous", "assortative")


@dataclass
class Product:
    """An infant immunisation product given at (or just before) birth."""

    coverage: float = 0.0  # proportion of eligible newborns protected
    efficacy_by_month: tuple[float, ...] = (0.0,) * PROTECTION_MONTHS
    seasonal_only: bool = False  # only infants born in the season window are given it
    season_start_day: int = 91  # 1 April (day of year)
    season_end_day: int = 273  # 30 September


def maternal_vaccine(coverage: float = 0.6, efficacy: float = 0.8) -> Product:
    """Maternal RSV vaccine with protection waning over months 4-6.

    Waning profile (0.8, 0.8, 0.8, 0.6, 0.4, 0.2 for months 1-6 at 80%
    efficacy) as used by Hogan et al. for Western Australia.
    """
    shape = (1.0, 1.0, 1.0, 0.75, 0.5, 0.25)
    return Product(coverage=coverage, efficacy_by_month=tuple(efficacy * s for s in shape))


def nirsevimab(coverage: float = 0.6, efficacy: float = 0.8, seasonal_only: bool = False) -> Product:
    """Long-acting monoclonal antibody, flat protection for about 6 months.

    HARMONIE reported 82.7% efficacy against RSV hospitalisation up to 180 days.
    """
    return Product(
        coverage=coverage,
        efficacy_by_month=(efficacy,) * PROTECTION_MONTHS,
        seasonal_only=seasonal_only,
    )


@dataclass
class Params:
    population: float = 1_000_000
    latent_days: float = 4.0
    infectious_days: float = 9.0
    immunity_days: float = 230.0
    r0: float = 1.6  # basic reproduction number (sets the transmission coefficient)
    seasonal_amplitude: float = 0.4  # b1
    peak_day: float = 100.0  # day of year with the highest transmission forcing (early April)
    mixing: str = "homogeneous"  # homogeneous | assortative
    # Relative susceptibility of infants (maternal antibodies), Hogan et al.
    sigma_month0: float = 0.08
    sigma_month1_2: float = 0.45
    # Relative infectiousness of people aged 10 years and over
    omega_10plus: float = 0.6
    # Proportion of infections hospitalised, by reporting group (Hogan et al., WA fit)
    hosp_prop: dict = field(
        default_factory=lambda: {
            "0-2 months": 0.424,
            "3-5 months": 0.088,
            "6-11 months": 0.047,
            "12-23 months": 0.020,
        }
    )

    def sigma(self) -> np.ndarray:
        s = np.ones(N_AGE)
        s[0] = self.sigma_month0
        s[1:3] = self.sigma_month1_2
        return s

    def omega(self) -> np.ndarray:
        w = np.ones(N_AGE)
        w[AGE_LOWER_YEARS >= 10] = self.omega_10plus
        return w

    def age_fractions(self) -> np.ndarray:
        # Uniform age distribution under constant births and cohort ageing.
        return AGE_WIDTHS_DAYS / LIFESPAN_DAYS

    def contact_matrix(self) -> np.ndarray:
        """Per-capita daily contact rates M[i, j]: contacts an i-person has with j-people.

        Homogeneous mixing gives M[i, j] = n_j / N. The assortative kernel adds
        strong within-age-band mixing and a parent-child band about 30 years
        apart, with total contacts symmetric (M[i, j] n_i == M[j, i] n_j). The
        overall scale is arbitrary because b0 is calibrated to R0.
        """
        if self.mixing not in MIXING_MODES:
            raise ValueError(f"mixing must be one of {MIXING_MODES}")
        n = self.age_fractions()
        if self.mixing == "homogeneous":
            return np.tile(n, (N_AGE, 1))
        a = AGE_MID_YEARS[:, None]
        b = AGE_MID_YEARS[None, :]
        gap = np.abs(a - b)
        kernel = 0.15 + np.exp(-((gap / 6.0) ** 2)) + 0.5 * np.exp(-(((gap - 30.0) / 8.0) ** 2))
        total = kernel * n[:, None] * n[None, :]  # symmetric total contact volume
        return total / n[:, None]

    def next_generation(self, beta: float = 1.0) -> np.ndarray:
        """K[i, j]: infections in age i caused by one infectious person in age j."""
        n = self.age_fractions()
        m = self.contact_matrix()
        m_ji = m.T  # contacts a j-person has with i-people
        return beta * self.infectious_days * self.sigma()[:, None] * m_ji * self.omega()[None, :]

    def b0(self) -> float:
        """Transmission coefficient giving the requested R0 (spectral radius of the next-generation matrix)."""
        rho = float(np.max(np.abs(np.linalg.eigvals(self.next_generation()))))
        return self.r0 / rho


# ---------------------------------------------------------------------------
# State vector layout
# ---------------------------------------------------------------------------
_S, _E, _I, _R = (slice(k * N_AGE, (k + 1) * N_AGE) for k in range(4))
_PM = slice(4 * N_AGE, 4 * N_AGE + PROTECTION_MONTHS)  # protected by maternal vaccine
_PN = slice(_PM.stop, _PM.stop + PROTECTION_MONTHS)  # protected by nirsevimab
_C = slice(_PN.stop, _PN.stop + N_MONTHLY)  # cumulative infections, monthly classes
N_STATE = _C.stop


def _initial_state(p: Params) -> np.ndarray:
    y = np.zeros(N_STATE)
    n = p.population * p.age_fractions()
    y[_I] = 1e-4 * n
    y[_S] = n - y[_I]
    return y


def _make_rhs(p: Params, maternal: Product | None, monoclonal: Product | None):
    sigma, omega = p.sigma(), p.omega()
    b0 = p.b0()
    contact = p.contact_matrix()
    n_by_age = p.population * p.age_fractions()
    eps, gam, delta = 1 / p.latent_days, 1 / p.infectious_days, 1 / p.immunity_days
    age_rate = 1 / AGE_WIDTHS_DAYS
    births_per_day = p.population / LIFESPAN_DAYS
    m_eff = np.array(maternal.efficacy_by_month) if maternal else np.zeros(PROTECTION_MONTHS)
    n_eff = np.array(monoclonal.efficacy_by_month) if monoclonal else np.zeros(PROTECTION_MONTHS)

    def in_season(prod: Product, t: float) -> bool:
        doy = t % DAYS_PER_YEAR
        return prod.season_start_day <= doy <= prod.season_end_day

    def rhs(t, y):
        S, E, I, R = y[_S], y[_E], y[_I], y[_R]
        PM, PN = y[_PM], y[_PN]
        beta = b0 * (1 + p.seasonal_amplitude * np.cos(2 * np.pi * (t - p.peak_day) / DAYS_PER_YEAR))
        lam = beta * (contact @ (omega * I / n_by_age))

        inf_S = sigma * lam * S
        lam_infant = lam[:PROTECTION_MONTHS]
        inf_PM = sigma[:PROTECTION_MONTHS] * (1 - m_eff) * lam_infant * PM
        inf_PN = sigma[:PROTECTION_MONTHS] * (1 - n_eff) * lam_infant * PN

        dy = np.zeros_like(y)
        dS, dE, dI, dR = dy[_S], dy[_E], dy[_I], dy[_R]
        dPM, dPN = dy[_PM], dy[_PN]

        # Epidemiological transitions
        dS += -inf_S + delta * R
        dE += inf_S - eps * E
        dE[:PROTECTION_MONTHS] += inf_PM + inf_PN
        dI += eps * E - gam * I
        dR += gam * I - delta * R
        dPM += -inf_PM
        dPN += -inf_PN

        # Ageing (outflow from every class; inflow from the class below)
        for comp, d in ((S, dS), (E, dE), (I, dI), (R, dR)):
            out = age_rate * comp
            d -= out
            d[1:] += out[:-1]
        for comp, d in ((PM, dPM), (PN, dPN)):
            out = age_rate[:PROTECTION_MONTHS] * comp
            d -= out
            d[1:] += out[:-1]
            dS[PROTECTION_MONTHS] += out[-1]  # protection ends after month 6

        # Births: split newborns between maternal, monoclonal and unprotected
        f_m = maternal.coverage if (maternal and (not maternal.seasonal_only or in_season(maternal, t))) else 0.0
        f_n = monoclonal.coverage if (monoclonal and (not monoclonal.seasonal_only or in_season(monoclonal, t))) else 0.0
        f_n *= 1 - f_m  # monoclonal offered to infants not protected by the maternal vaccine
        dPM[0] += births_per_day * f_m
        dPN[0] += births_per_day * f_n
        dS[0] += births_per_day * (1 - f_m - f_n)

        # Cumulative infections in monthly classes (all susceptible pathways)
        cum = inf_S[:N_MONTHLY].copy()
        cum[:PROTECTION_MONTHS] += inf_PM + inf_PN
        dy[_C] = cum
        return dy

    return rhs


def _rhs_core(t, y, b0, amp, peak, sigma, omega, contact, n_by_age, eps, gam, delta, age_rate, births):
    """The right-hand side without immunisation products, written as loops for Numba.

    Identical to the closure in :func:`_make_rhs` with ``maternal`` and
    ``monoclonal`` absent; ``tests/test_fit.py`` checks the two agree.
    """
    n = N_AGE
    dy = np.zeros_like(y)
    beta = b0 * (1.0 + amp * np.cos(2.0 * np.pi * (t - peak) / DAYS_PER_YEAR))
    lam = np.empty(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += contact[i, j] * omega[j] * y[2 * n + j] / n_by_age[j]
        lam[i] = beta * acc
    for i in range(n):
        s_, e_, i_, r_ = y[i], y[n + i], y[2 * n + i], y[3 * n + i]
        inf_s = sigma[i] * lam[i] * s_
        dy[i] += -inf_s + delta * r_
        dy[n + i] += inf_s - eps * e_
        dy[2 * n + i] += eps * e_ - gam * i_
        dy[3 * n + i] += gam * i_ - delta * r_
        for k in range(4):
            out = age_rate[i] * y[k * n + i]
            dy[k * n + i] -= out
            if i + 1 < n:
                dy[k * n + i + 1] += out
        if i < N_MONTHLY:
            dy[4 * n + 2 * PROTECTION_MONTHS + i] = inf_s
    pm0 = 4 * n
    pn0 = 4 * n + PROTECTION_MONTHS
    c0 = pn0 + PROTECTION_MONTHS
    for i in range(PROTECTION_MONTHS):
        pm, pn = y[pm0 + i], y[pn0 + i]
        inf_pm = sigma[i] * lam[i] * pm
        inf_pn = sigma[i] * lam[i] * pn
        dy[n + i] += inf_pm + inf_pn
        dy[pm0 + i] -= inf_pm
        dy[pn0 + i] -= inf_pn
        out_m = age_rate[i] * pm
        out_n = age_rate[i] * pn
        dy[pm0 + i] -= out_m
        dy[pn0 + i] -= out_n
        if i + 1 < PROTECTION_MONTHS:
            dy[pm0 + i + 1] += out_m
            dy[pn0 + i + 1] += out_n
        else:
            dy[PROTECTION_MONTHS] += out_m + out_n
        dy[c0 + i] += inf_pm + inf_pn
    dy[0] += births
    return dy


if numba is not None:
    _rhs_core = numba.njit(cache=True)(_rhs_core)


def _make_rhs_compiled(p: Params):
    """Compiled right-hand side for runs without immunisation products."""
    sigma, omega = p.sigma(), p.omega()
    b0 = p.b0()
    contact = np.ascontiguousarray(p.contact_matrix())
    n_by_age = p.population * p.age_fractions()
    eps, gam, delta = 1 / p.latent_days, 1 / p.infectious_days, 1 / p.immunity_days
    age_rate = 1 / AGE_WIDTHS_DAYS
    births_per_day = p.population / LIFESPAN_DAYS
    amp, peak = float(p.seasonal_amplitude), float(p.peak_day)

    def rhs(t, y):
        return _rhs_core(t, y, b0, amp, peak, sigma, omega, contact, n_by_age, eps, gam, delta, age_rate, births_per_day)

    return rhs


# scipy's LSODA wrapper is not re-entrant: two solves on different threads raise
# IntegratorConcurrencyError. The API runs requests on a thread pool, so serialise.
_SOLVER_LOCK = threading.Lock()


def _solve(p: Params, y0, t0, t1, maternal=None, monoclonal=None, step=7.0):
    t_eval = np.arange(t0, t1 + 1e-9, step)
    with _SOLVER_LOCK:
        sol = _integrate(p, y0, t0, t1, t_eval, maternal, monoclonal)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


def _integrate(p, y0, t0, t1, t_eval, maternal, monoclonal):
    compiled = numba is not None and maternal is None and monoclonal is None
    return solve_ivp(
        _make_rhs_compiled(p) if compiled else _make_rhs(p, maternal, monoclonal),
        (t0, t1),
        y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-6,
        max_step=5.0,
    )


@lru_cache(maxsize=64)
def _burn_in(params_key: tuple, years: int):
    p = Params(**dict(params_key))
    t, y = _solve(p, _initial_state(p), 0.0, years * DAYS_PER_YEAR, step=30.0)
    return y[:, -1].copy()


def _params_key(p: Params) -> tuple:
    d = asdict(p)
    d.pop("hosp_prop")
    return tuple(sorted(d.items()))


def _hospitalisations(p: Params, y: np.ndarray) -> dict[str, np.ndarray]:
    """Weekly hospitalisations by reporting group from cumulative infections."""
    weekly_inf = np.diff(y[_C], axis=1)  # infections per step, by monthly class
    return {g: p.hosp_prop[g] * weekly_inf[list(idx)].sum(axis=0) for g, idx in HOSP_GROUPS.items()}


def simulate(
    params: Params | None = None,
    maternal: Product | None = None,
    monoclonal: Product | None = None,
    years: int = 5,
    burn_in_years: int = 40,
) -> dict:
    """Run baseline and intervention scenarios from the same endemic state.

    Returns weekly series and annual summaries for both runs.
    """
    p = params or Params()
    y_start = _burn_in(_params_key(p), burn_in_years)
    t0 = burn_in_years * DAYS_PER_YEAR
    t1 = t0 + years * DAYS_PER_YEAR

    y_start = y_start.copy()
    y_start[_C] = 0.0
    t, y_base = _solve(p, y_start, t0, t1)
    _, y_int = _solve(p, y_start, t0, t1, maternal, monoclonal)

    h_base = _hospitalisations(p, y_base)
    h_int = _hospitalisations(p, y_int)
    infant_inf_base = np.diff(y_base[_C], axis=1).sum(axis=0)
    infant_inf_int = np.diff(y_int[_C], axis=1).sum(axis=0)
    total_prev_base = y_base[_I].sum(axis=0)

    summary = []
    for g in HOSP_GROUPS:
        b, s = h_base[g].sum() / years, h_int[g].sum() / years
        summary.append(
            {
                "group": g,
                "baseline_per_year": float(b),
                "scenario_per_year": float(s),
                "averted_per_year": float(b - s),
                "percent_averted": float(100 * (b - s) / b) if b > 0 else 0.0,
            }
        )
    tot_b = sum(r["baseline_per_year"] for r in summary)
    tot_s = sum(r["scenario_per_year"] for r in summary)
    summary.append(
        {
            "group": "Under 2 years",
            "baseline_per_year": tot_b,
            "scenario_per_year": tot_s,
            "averted_per_year": tot_b - tot_s,
            "percent_averted": 100 * (tot_b - tot_s) / tot_b if tot_b > 0 else 0.0,
        }
    )

    weeks = (t[1:] - t0) / 7.0
    births_per_year = p.population / LIFESPAN_DAYS * DAYS_PER_YEAR
    doses = _doses_per_year(births_per_year, maternal, monoclonal)
    averted = tot_b - tot_s
    under6_b = h_base["0-2 months"] + h_base["3-5 months"]
    under6_s = h_int["0-2 months"] + h_int["3-5 months"]
    return {
        "doses": {
            **doses,
            "per_admission_averted": float(doses["total"] / averted) if averted > 1e-9 else None,
        },
        "peak_under6": {
            "baseline": float(under6_b.max()),
            "scenario": float(under6_s.max()),
            "baseline_week": float(weeks[int(under6_b.argmax())]),
            "scenario_week": float(weeks[int(under6_s.argmax())]),
        },
        "weeks": weeks.tolist(),
        "day_of_year": ((t[1:]) % DAYS_PER_YEAR).tolist(),
        "hospitalisations": {
            "baseline": {g: v.tolist() for g, v in h_base.items()},
            "scenario": {g: v.tolist() for g, v in h_int.items()},
        },
        "infant_infections": {"baseline": infant_inf_base.tolist(), "scenario": infant_inf_int.tolist()},
        "prevalence_all_ages": total_prev_base[1:].tolist(),
        "summary": summary,
        "params": {**asdict(p), "b0": p.b0()},
        "births_per_year": births_per_year,
    }


def _season_fraction(prod: Product) -> float:
    return (prod.season_end_day - prod.season_start_day + 1) / DAYS_PER_YEAR


def _doses_per_year(births_per_year: float, maternal: Product | None, monoclonal: Product | None) -> dict:
    """Average yearly doses implied by the birth flows in the model."""
    f_m = maternal.coverage * (_season_fraction(maternal) if maternal.seasonal_only else 1.0) if maternal else 0.0
    f_n = 0.0
    if monoclonal:
        season = _season_fraction(monoclonal) if monoclonal.seasonal_only else 1.0
        # The monoclonal is offered to infants not already covered by the maternal vaccine.
        f_n = monoclonal.coverage * season * (1 - (maternal.coverage if maternal and not maternal.seasonal_only else 0.0))
    m, n = births_per_year * f_m, births_per_year * f_n
    return {"maternal": float(m), "nirsevimab": float(n), "total": float(m + n)}
