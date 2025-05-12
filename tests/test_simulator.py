"""
Unit tests for функцию simulate_statistics из src.simulator.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulator import simulate_statistics


def test_simulate_statistics_knn_shape_and_cols():
    """Для knn-режима DataFrame имеет форму (trials×4), avg_clique == NaN."""
    df0, df1 = simulate_statistics(n=8, trials=6, graph_type="knn", param=2, beta=1.0)
    assert isinstance(df0, pd.DataFrame) and isinstance(df1, pd.DataFrame)
    assert df0.shape == (6, 4) and df1.shape == (6, 4)
    assert df0["avg_clique"].isna().all() and df1["avg_clique"].isna().all()


def test_simulate_statistics_dist_clique_not_nan():
    """Для dist-режима avg_clique должно быть вычислено."""
    df0, _ = simulate_statistics(n=8, trials=6, graph_type="dist", param=0.3, beta=1.0)
    assert not df0["avg_clique"].isna().any()


def test_simulate_statistics_values_non_negative():
    """num_edges, num_triangles, max_degree ≥ 0 всегда."""
    df0, df1 = simulate_statistics(10, trials=5, graph_type="knn", param=1, beta=1.0)
    for col in ["num_edges", "num_triangles", "max_degree"]:
        assert (df0[col] >= 0).all() and (df1[col] >= 0).all()


def test_simulate_statistics_reproducible_with_seed():
    """Установка одного и того же seed даёт одни и те же DataFrame."""
    np.random.seed(42)
    a0, a1 = simulate_statistics(5, trials=3, graph_type="knn", param=1, beta=1.0)
    np.random.seed(42)
    b0, b1 = simulate_statistics(5, trials=3, graph_type="knn", param=1, beta=1.0)
    pd.testing.assert_frame_equal(a0, b0)
    pd.testing.assert_frame_equal(a1, b1)


def test_simulate_statistics_invalid_inputs_raise():
    """Плохие входные параметры → ValueError."""
    with pytest.raises(ValueError):
        simulate_statistics(n=-1, trials=3, graph_type="knn", param=1, beta=1.0)
    with pytest.raises(ValueError):
        simulate_statistics(n=5, trials=3, graph_type="unknown", param=1, beta=1.0)
