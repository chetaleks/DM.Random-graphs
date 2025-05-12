"""
Unit tests for функции из модуля src.graphs.
"""

import numpy as np
import pytest

from src.graphs import build_distance_graph, build_knn_graph


def test_distance_graph_basic():
    """Простейший distance-граф."""
    data = np.array([0.0, 0.5, 2.0, 2.4])
    graph = build_distance_graph(data, d=1.0)
    edges = set(graph.edges())
    assert (0, 1) in edges and (2, 3) in edges
    assert graph.number_of_edges() == 2


def test_distance_graph_no_edges():
    """Нет ребер, если все точки далеко."""
    graph = build_distance_graph(np.array([0, 10, 20]), d=1.0)
    assert graph.number_of_edges() == 0


def test_distance_graph_negative_d_raises():
    """Отрицательный параметр d → ValueError."""
    with pytest.raises(ValueError):
        build_distance_graph(np.array([0.0, 1.0]), d=-1.0)


def test_knn_graph_minimal():
    """KNN-граф с k=1 на трёх точках даёт 2 ребра."""
    data = np.array([0.0, 1.0, 10.0])
    graph = build_knn_graph(data, k=1)
    assert graph.number_of_edges() == 2
    assert set(graph.nodes()) == {0, 1, 2}


def test_knn_invalid_k_zero_and_large():
    """k=0 или k>=n → ValueError."""
    data = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        build_knn_graph(data, k=0)
    with pytest.raises(ValueError):
        build_knn_graph(data, k=2)


def test_knn_no_self_loops_and_correct_edges():
    """Убедимся, что нет петель и связь правильная."""
    data = np.array([0.0, 1.0, 2.0, 3.0])
    graph = build_knn_graph(data, k=1)
    assert all(u != v for u, v in graph.edges())
    expected = {(0, 1), (1, 2), (2, 3)}
    assert set(map(tuple, map(sorted, graph.edges()))) == expected
