import numpy as np
import pytest

from src.mmm import apply_adstock


def test_adstock_known_values_decay_half():
    """Hand-computed geometric adstock with decay=0.5 on a constant spend series."""
    spend = np.array([10.0, 10.0, 10.0, 10.0])
    result = apply_adstock(spend, decay_rate=0.5)
    # adstock[0] = 10
    # adstock[1] = 10 + 0.5*10   = 15
    # adstock[2] = 10 + 0.5*15   = 17.5
    # adstock[3] = 10 + 0.5*17.5 = 18.75
    expected = np.array([10.0, 15.0, 17.5, 18.75])
    np.testing.assert_allclose(result, expected)


def test_adstock_zero_decay_is_identity():
    spend = np.array([5.0, 12.0, 0.0, 30.0])
    result = apply_adstock(spend, decay_rate=0.0)
    np.testing.assert_allclose(result, spend)


def test_adstock_single_impulse_decays_geometrically():
    """A single spend impulse should decay as decay_rate^t in later periods."""
    spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    decay = 0.6
    result = apply_adstock(spend, decay_rate=decay)
    expected = 100.0 * decay ** np.arange(5)
    np.testing.assert_allclose(result, expected)


def test_adstock_increases_or_equal_vs_raw_spend_when_decay_positive():
    spend = np.array([10.0, 10.0, 10.0])
    result = apply_adstock(spend, decay_rate=0.4)
    assert np.all(result >= spend - 1e-9)


@pytest.mark.parametrize("bad_decay", [-0.1, 1.0, 1.5])
def test_adstock_rejects_invalid_decay(bad_decay):
    with pytest.raises(ValueError):
        apply_adstock(np.array([1.0, 2.0, 3.0]), decay_rate=bad_decay)
