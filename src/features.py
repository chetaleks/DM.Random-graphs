"""
Модуль features: функции для вычисления характеристик графов.
"""

import networkx as nx
import numpy as np


def compute_triangles(graph: nx.Graph) -> int:
    """
    Возвращает число треугольников в неориентированном графе.

    Триангулы считаются суммой треугольников на каждой вершине делённой на 3.
    """
    return sum(nx.triangles(graph).values()) // 3


def compute_edge_count(graph: nx.Graph) -> int:
    """
    Возвращает число рёбер графа.
    """
    return graph.number_of_edges()


def compute_clique_number_1d(data: np.ndarray, d: float) -> int:
    """
    Кликовое число ω(G) для distance-графа на прямой (1D):
    максимальная длина подпоследовательности точек,
    попадающих в окно ширины d.
    """
    n = len(data)
    if n == 0:
        raise IndexError("Empty data")
    x = np.sort(data)
    max_clique = 0
    j = 0
    for i in range(n):
        while j < n and x[j] - x[i] <= d:
            j += 1
        size = j - i
        # замена if size>max_clique на встроенный max
        max_clique = max(max_clique, size)
    return max_clique
def count_connected_components(graph):
    """
    Вычисляет число компонент связности в неориентированном графе.

    :param graph: Словарь, представляющий граф в виде смежности.
                  Ключи - вершины, значения - списки соседей.
    :return: Число компонент связности в графе.
    """
    visited = set()  # Множество для отслеживания посещенных вершин
    component_count = 0  # Счетчик компонент связности

    def dfs(vertex):
        """Вспомогательная функция для выполнения DFS."""
        stack = [vertex]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    for vertex in graph:
        if vertex not in visited:
            # Если вершина не посещена, это новая компонента связности
            visited.add(vertex)
            dfs(vertex)
            component_count += 1

    return component_count




# Функция для подсчета хроматического числа
def compute_chromatic_number(G):
    """Возвращает хроматическое число χ(G) графа G."""
    # Используем жадную раскраску
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    chromatic_number = len(set(coloring.values()))  # Количество уникальных цветов
    return chromatic_number
