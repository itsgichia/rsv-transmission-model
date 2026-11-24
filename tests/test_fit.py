"""Fast tests for the calibration module. Sampling itself is exercised by
scripts/fit_recovery.py, which is too slow for the unit suite."""

import numpy as np
import pytest

from rsvmodel.fit import (
    merge_groups,
    save_netcdf,
    DEFAULT_BANDS,
    AgeBand,
    FitParams,
    Observations,
    band_matrix,
    expected_counts,
    log_likelihood,
    monthly_incidence,
    negative_binomial_logpmf,
    simulate_observations,
)
from rsvmodel.model import N_AGE

TRUTH = FitParams(1.6, 0.4, 100.0, 230.0)
RHO = np.array([0.30, 0.15, 0.08, 0.03, 0.01, 0.005, 0.005, 0.01, 0.03])


@pytest.fixture(scope="module")
def synthetic():
    return simulate_observations(TRUTH, RHO, 20.0, 2023, 1, 24, 8_300_000, seed=3, burn_in_years=15)


def test_default_bands_partition_every_model_class():
    matrix = band_matrix(DEFAULT_BANDS)
    assert matrix.shape == (len(DEFAULT_BANDS), N_AGE)
    assert np.allclose(matrix.sum(axis=0), 1.0)


def test_coarse_band_splits_a_model_class_proportionally():
    (row,) = band_matrix([AgeBand("under 3 years", 0.0, 3.0)])
    assert np.allclose(row[:24], 1.0)  # all 24 monthly classes
    assert row[24] == pytest.approx(1 / 3)  # one third of the 2-4 year class
    assert np.all(row[25:] == 0)


def test_bad_band_rejected():
    with pytest.raises(ValueError):
        AgeBand("backwards", 5.0, 2.0)


def test_month_labels_roll_over_years():
    obs = Observations(2024, 11, DEFAULT_BANDS, np.zeros((9, 4)), 1.0)
    assert obs.month_labels() == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_incidence_is_positive_and_seasonal(synthetic):
    inc = monthly_incidence(TRUTH, synthetic, burn_in_years=15)
    assert inc.shape == (9, 24)
    assert (inc > 0).all()
    total = inc.sum(axis=0).reshape(2, 12)
    peak_month = total.mean(axis=0).argmax() + 1
    assert 5 <= peak_month <= 8, "infections should peak in winter"


def test_negative_binomial_matches_scipy():
    from scipy.stats import nbinom

    mu, alpha, k = np.array([3.0, 40.0]), 7.0, np.array([2, 55])
    ours = negative_binomial_logpmf(k, mu, alpha)
    theirs = nbinom.logpmf(k, alpha, alpha / (alpha + mu))
    assert np.allclose(ours, theirs)


def test_likelihood_prefers_truth_over_perturbations(synthetic):
    ll_truth = log_likelihood(TRUTH, RHO, 20.0, synthetic, burn_in_years=15)
    for worse in (
        FitParams(1.45, 0.4, 100.0, 230.0),
        FitParams(1.6, 0.25, 100.0, 230.0),
        FitParams(1.6, 0.4, 150.0, 230.0),
        FitParams(1.6, 0.4, 100.0, 150.0),
    ):
        assert log_likelihood(worse, RHO, 20.0, synthetic, burn_in_years=15) < ll_truth


def test_expected_counts_scale_with_reporting_fraction(synthetic):
    inc = monthly_incidence(TRUTH, synthetic, burn_in_years=15)
    half = expected_counts(inc, RHO / 2)
    assert np.allclose(half * 2, expected_counts(inc, RHO))


def test_csv_round_trip(tmp_path, synthetic):
    import csv

    path = tmp_path / "obs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "age_band", "lower_years", "upper_years", "count"])
        for band, row in zip(synthetic.bands, synthetic.counts):
            for month, count in zip(synthetic.month_labels(), row):
                writer.writerow([month, band.label, band.lower_years, band.upper_years, int(count)])
    loaded = Observations.from_csv(path, population=synthetic.population)
    assert loaded.month_labels() == synthetic.month_labels()
    assert [b.label for b in loaded.bands] == [b.label for b in synthetic.bands]
    assert np.array_equal(loaded.counts, synthetic.counts)


def test_merge_groups_adds_only_missing_groups():
    xr = pytest.importorskip("xarray")
    post = xr.Dataset({"r0": ("draw", np.array([1.5, 1.6]))})
    tree = xr.DataTree.from_dict({"posterior": post})
    other = xr.DataTree.from_dict({
        "posterior": xr.Dataset({"r0": ("draw", np.array([9.0, 9.0]))}),
        "prior_predictive": xr.Dataset({"counts": ("draw", np.array([3, 4]))}),
    })
    merged = merge_groups(tree, other)
    assert set(merged.children) == {"posterior", "prior_predictive"}
    np.testing.assert_allclose(merged["posterior"].to_dataset()["r0"].values, [1.5, 1.6])
    assert merge_groups(merged, other) is merged


def test_save_netcdf_drops_ragged_keeps_string_coords(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    ragged = np.empty(2, dtype=object)
    ragged[0], ragged[1] = [1.0, 2.0], [1.0, 2.0, 3.0]
    tree = xr.DataTree.from_dict({
        "posterior": xr.Dataset({"rho": (("band",), np.array([0.1, 0.2]))}, coords={"band": np.array(["0-5m", "6-11m"], dtype=object)}),
        "sample_stats": xr.Dataset({"log_marginal_likelihood": (("chain",), ragged), "beta": (("chain",), np.array([1.0, 1.0]))}),
    })
    tree.attrs["population"] = 8.3e6
    dropped = save_netcdf(tree, tmp_path / "x.nc")
    assert dropped == ["sample_stats/log_marginal_likelihood"]
    back = xr.open_datatree(tmp_path / "x.nc")
    assert list(back["posterior"].to_dataset()["band"].values) == ["0-5m", "6-11m"]
    assert "beta" in back["sample_stats"].to_dataset()
    assert back.attrs["population"] == 8.3e6
    assert back.attrs["dropped_for_netcdf"] == "sample_stats/log_marginal_likelihood"


def test_window_slices_consecutive_months(synthetic):
    sub = synthetic.window("2023-03", "2023-08")
    assert sub.month_labels() == ["2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08"]
    assert sub.counts.shape == (len(synthetic.bands), 6)
    np.testing.assert_array_equal(sub.counts[:, 0], synthetic.counts[:, 2])
    assert synthetic.window(None, None).month_labels() == synthetic.month_labels()
    with pytest.raises(ValueError):
        synthetic.window("2023-08", "2023-03")
    with pytest.raises(ValueError):
        synthetic.window("1999-01")
