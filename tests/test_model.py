import numpy as np
import pytest

from rsvmodel import Params, maternal_vaccine, nirsevimab, simulate
from rsvmodel.model import (
    N_STATE, _C, _burn_in, _params_key, _initial_state, _make_rhs, _solve, DAYS_PER_YEAR,
)


def total_population(y):
    return y[: _C.start].sum(axis=0)


def test_population_is_conserved():
    p = Params()
    t, y = _solve(p, _initial_state(p), 0, 5 * DAYS_PER_YEAR, maternal_vaccine(0.7), nirsevimab(0.5))
    pop = total_population(y)
    assert np.allclose(pop, p.population, rtol=1e-4)


def test_no_intervention_matches_baseline():
    r = simulate(Params(), None, None, years=2)
    for row in r["summary"]:
        assert row["averted_per_year"] == pytest.approx(0, abs=1e-6)


def test_states_stay_non_negative():
    p = Params()
    t, y = _solve(p, _initial_state(p), 0, 10 * DAYS_PER_YEAR)
    assert (y > -1e-6).all()


def test_more_coverage_averts_more():
    low = simulate(Params(), maternal_vaccine(0.3), None, years=3)["summary"][-1]["percent_averted"]
    high = simulate(Params(), maternal_vaccine(0.8), None, years=3)["summary"][-1]["percent_averted"]
    assert high > low > 0


def test_youngest_infants_benefit_most_from_maternal_vaccine():
    s = {r["group"]: r["percent_averted"] for r in simulate(Params(), maternal_vaccine(0.6), None, years=3)["summary"]}
    assert s["0-2 months"] > s["3-5 months"] > s["12-23 months"]


def test_r0_sets_transmission_coefficient():
    assert Params(r0=2.0).b0() == pytest.approx(2 * Params(r0=1.0).b0())


def test_contact_matrix_is_balanced_and_r0_is_respected():
    """Total contacts are symmetric and the next-generation matrix has spectral radius R0."""
    for mixing in ("homogeneous", "assortative"):
        p = Params(r0=1.7, mixing=mixing)
        n = p.age_fractions()
        m = p.contact_matrix()
        assert np.allclose(m * n[:, None], (m * n[:, None]).T)
        rho = max(abs(np.linalg.eigvals(p.next_generation(p.b0()))))
        assert rho == pytest.approx(1.7)


def test_assortative_mixing_changes_dynamics_but_not_population():
    hom = simulate(Params(mixing="homogeneous"), nirsevimab(0.6), years=2)
    ass = simulate(Params(mixing="assortative"), nirsevimab(0.6), years=2)
    assert hom["summary"][-1]["baseline_per_year"] != pytest.approx(ass["summary"][-1]["baseline_per_year"], rel=1e-3)
    assert ass["summary"][-1]["percent_averted"] > 0


def test_unknown_mixing_mode_is_rejected():
    with pytest.raises(ValueError):
        Params(mixing="random").contact_matrix()


def test_seasonal_only_programme_uses_fewer_doses_and_averts_less():
    full = simulate(Params(), monoclonal=nirsevimab(0.6), years=3)
    season = simulate(Params(), monoclonal=nirsevimab(0.6, seasonal_only=True), years=3)
    assert season["doses"]["total"] < full["doses"]["total"]
    assert season["summary"][-1]["percent_averted"] < full["summary"][-1]["percent_averted"]


def test_cli_runs(capsys):
    from rsvmodel.cli import main

    assert main(["run", "--maternal", "0.6", "--years", "1", "--format", "csv"]) == 0
    out = capsys.readouterr().out
    assert "Under 2 years" in out and out.startswith("group,")
    assert main(["defaults"]) == 0
