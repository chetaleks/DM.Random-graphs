[![CI](https://github.com/chetaleks/DM.Random-graphs/actions/workflows/CI.yml/badge.svg)](https://github.com/chetaleks/DM.Random-graphs/actions/workflows/CI.yml)  

# Анализ числовых характеристик случайных графов

## Описание проекта
В рамках этого проекта мы реализуем инструменты и проводим серию экспериментов для исследования различных числовых характеристик случайных графов, построенных на точках из разных распределений.  
Главные задачи:
- Генерация выборок (`src/generators.py`)
- Построение KNN- и distance-графов (`src/graphs.py`)
- Вычисление признаков (`src/features.py`)
- Проведение Monte-Carlo экспериментов и визуализация

## Быстрый старт

1. **Клонировать репозиторий**
   ```bash
   git clone https://github.com/chetaleks/DM.Random-graphs.git
   cd DM.Random-graphs
    ````

2. **Создать и активировать виртуальное окружение**

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux/macOS
   .venv\Scripts\activate        # Windows
   ```

3. **Установить зависимости**

   ```bash
   pip install -r requirements.txt
   ```

4. **Запустить тесты, линтер и форматтер**

   ```bash
   pytest
   flake8
   black --check src tests notebooks report
   ```

5. **Открыть Jupyter-ноутбук**

   ```bash
   jupyter lab notebooks/part1_experiments.ipynb
   ```

## Пример использования

```python
from src.simulator import simulate_statistics

df0, df1 = simulate_statistics(
    n=200,
    trials=100,
    graph_type="knn",
    param=5,
    beta=1.0
)
print(df0.head())
```

## Как помочь проекту

* Открыть [issue](https://github.com/chetaleks/DM.Random-graphs/issues)
* Форкнуть репозиторий, создать ветку `yourname/feature`, внести изменения и сделать Pull Request
* Следить за CI и исправлять замечания линтера


## Структура репозитория

```
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── generators.py
│   ├── graphs.py
│   ├── features.py
│   └── simulator.py
├── notebooks/
│   └── part1_experiments.ipynb
├── report/
│   ├── report.tex
│   └── report.pdf
└── tests/
    ├── test_generators.py
    ├── test_graphs.py
    ├── test_features.py
    └── test_simulator.py
```

## Лицензия

MIT © 2025 chetaleks

