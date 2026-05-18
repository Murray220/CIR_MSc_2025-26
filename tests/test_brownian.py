import numpy as np
import pytest

from src.utils.brownian import aggregate_brownian_increments
from src.utils.rng import make_brownian_increments, make_rng

def test_aggregate_brownian_increments_have_correct_shape():
    dW_fine = np.ones((5, 12))
    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 3)

    assert dW_coarse.shape == (5,4)

def test_aggregate_brownian_increments_sum_blocks_correctly():
    dW_fine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 0.5, 1.0, 1.0, 2.0, 2.0],
        ]
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 2)

    expected = np.array(
        [
            [3.0, 7.0, 11.0],
            [1.0, 2.0, 4.0],
        ]
    )

    np.testing.assert_allclose(dW_coarse, expected)

def test_aggregate_brownian_increments_with_factor_one_return_same():
    dW_fine = np.array(
        [
            [1.0, -2.0, 3.0]
        ]
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 1)

    np.testing.assert_allclose(dW_coarse, dW_fine)

def test_aggregate_brownian_rejects_negative_factor():
    dW_fine = np.ones(
        (2,4)
    )

    with pytest.raises(ValueError, match = "factor must be positive"):
        aggregate_brownian_increments(dW_fine, factor = 0)

def test_aggregate_brownian_increments_reject_non2darray():
    dW_fine = np.ones(8)

    with pytest.raises(ValueError, match = "dW_fine needs to be a 2D array"):
        aggregate_brownian_increments(dW_fine, factor = 2)

def test_aggregate_brownian_increments_rejects_nondivisible_factor():
    dW_fine = np.ones(
        (2,5)
    )

    with pytest.raises(ValueError, match = "number of fine steps must be divisible by chosen factor"):
        aggregate_brownian_increments(dW_fine, factor = 2)

def test_aggregate_brownian_increments_has_correct_total():
    rng = make_rng(123)
    dt_fine = 1.0 / 16

    dW_fine = make_brownian_increments(
        rng = rng,
        n_paths = 100,
        n_steps = 16,
        dt = dt_fine,
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = 4)

    fine_total = dW_fine.sum(axis = 1)
    coarse_total = dW_coarse.sum(axis = 1)

    np.testing.assert_allclose(coarse_total, fine_total)

def test_aggregate_brownian_increments_have_correct_var_scale():
    rng = make_rng(123)

    n_paths = 50_000
    n_fine_steps = 16
    factor = 4
    dt_fine = 1.0 / n_fine_steps
    dt_coarse = factor * dt_fine

    dW_fine = make_brownian_increments(
        rng = rng,
        n_paths = n_paths,
        n_steps = n_fine_steps,
        dt = dt_fine,
    )

    dW_coarse = aggregate_brownian_increments(dW_fine, factor = factor)

    empirical_variance = np.var(dW_coarse[:, 0], ddof = 1)

    assert abs(empirical_variance - dt_coarse) < 0.01
