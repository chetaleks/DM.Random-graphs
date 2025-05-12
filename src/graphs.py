"""
Модуль graphs: функции для построения KNN- и distance-графов.
"""

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


def build_knn_graph(data: np.ndarray, k: int) -> nx.Graph:
    """
    Строит KNN-граф: каждая вершина соединена с k ближайшими соседями.
    """
    n = len(data)
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")
    tree = KDTree(data.reshape(-1, 1))
    _, neighbors = tree.query(data.reshape(-1, 1), k=k + 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i != j:
                graph.add_edge(i, j)
    return graph


def build_distance_graph(data: np.ndarray, d: float) -> nx.Graph:
    """
    Строит distance-граф: ребро между i и j, если |data[i]-data[j]| <= d.
    """
    if d < 0:
        raise ValueError("Distance threshold d must be non-negative")
    graph = nx.Graph()
    n = len(data)
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(data[i] - data[j]) <= d:
                graph.add_edge(i, j)
    return graph
