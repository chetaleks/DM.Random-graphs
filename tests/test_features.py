"""
Unit tests for функции из модуля src.features.
"""

import networkx as nx
import numpy as np
import pytest

from src.features import (
    compute_clique_number_1d,
    compute_edge_count,
    compute_triangles,
)


def test_compute_triangles_empty():
    """Пустой граф не содержит треугольников."""
    graph = nx.Graph()
    assert compute_triangles(graph) == 0


def test_compute_triangles_complete_graph():
    """В полном графе на 4 вершинах ровно 4 треугольника."""
    graph = nx.complete_graph(4)
    assert compute_triangles(graph) == 4


def test_compute_edge_count_path_graph():
    """В пути из 5 вершин ровно 4 ребра."""
    graph = nx.path_graph(5)
    assert compute_edge_count(graph) == 4


def test_compute_edge_count_empty_graph():
    """Пустой граф имеет 0 ребер."""
    graph = nx.Graph()
    assert compute_edge_count(graph) == 0


def test_compute_clique_number_1d_various():
    """Проверяем разные случаи для 1D-кликового числа."""
    data = np.array([0, 1, 2, 5, 7])
    assert compute_clique_number_1d(data, d=2) == 3
    assert compute_clique_number_1d(data, d=5) == 4
    assert compute_clique_number_1d(data, d=0) == 1


def test_compute_clique_number_1d_single_and_identical():
    """Одиночный и все совпадающие точки."""
    assert compute_clique_number_1d(np.array([42.0]), d=10) == 1
    data = np.array([3.14] * 5)
    assert compute_clique_number_1d(data, d=0) == 5


def test_compute_clique_number_1d_empty_raises():
    """Пустой массив вызывает IndexError."""
    with pytest.raises(IndexError):
        compute_clique_number_1d(np.array([]), d=1.0)
