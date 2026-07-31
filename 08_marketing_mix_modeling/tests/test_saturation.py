import numpy as np
import pytest

from src.mmm import apply_saturation


def test_saturation_zero_input_is_zero():
    assert apply_saturation(np.array([0.0]), gamma=100.0)[0] == pytest.approx(0.0)


def test_saturation_half_point_equals_half():
    """By construction f(gamma) == 0.5 for alpha == 1."""
    gamma = 250.0
    value = apply_saturation(np.array([gamma]), gamma=gamma, alpha=1.0)[0]
    assert value == pytest.approx(0.5, abs=1e-9)


def test_saturation_is_monotonically_non_decreasing():
    x = np.linspace(0, 100_000, 500)
    y = apply_saturation(x, gamma=10_000, alpha=1.3)
    diffs = np.diff(y)
    assert np.all(diffs >= -1e-12)


def test_saturation_bounded_between_zero_and_one():
    x = np.array([0.0, 1.0, 100.0, 10_000.0, 1_000_000.0, 1e9])
    y = apply_saturation(x, gamma=5_000.0)
    assert np.all(y >= 0.0)
    assert np.all(y < 1.0)


def test_saturation_approaches_one_for_very_large_spend():
    y = apply_saturation(np.array([1e12]), gamma=1000.0)[0]
    assert y > 0.999


def test_saturation_diminishing_returns_shape():
    """Marginal contribution (derivative) should shrink as spend grows."""
    x = np.linspace(1, 100_000, 1000)
    y = apply_saturation(x, gamma=20_000, alpha=1.0)
    marginal = np.diff(y) / np.diff(x)
    # Later marginal returns should be smaller than earlier ones.
    assert marginal[10] > marginal[-10]


@pytest.mark.parametrize("bad_gamma", [0.0, -5.0])
def test_saturation_rejects_non_positive_gamma(bad_gamma):
    with pytest.raises(ValueError):
        apply_saturation(np.array([1.0, 2.0]), gamma=bad_gamma)


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
def test_saturation_rejects_non_positive_alpha(bad_alpha):
    with pytest.raises(ValueError):
        apply_saturation(np.array([1.0, 2.0]), gamma=10.0, alpha=bad_alpha)
