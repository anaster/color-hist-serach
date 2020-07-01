import numpy as np

# Класс движка поисковой системы:
# Сравниваниет параметры индексированного датасета
# и сравнивает с параметрами изображения-образца
# через расчет метрики
class Searcher:
    def __init__(self, index):
        self.index = index

    def search(self, queryFeatures):
        results = {}
        for (k, features) in self.index.items():
            d = self.chi2_distance(features, queryFeatures)
            results[k] = d
        # Результаты сортируются по возрастанию метрики
        results = sorted([(v, k) for (k, v) in results.items()])
        return results
    # Расчет метрики хи-квадрат, добалвление малого значение
    # eps решает проблему деления на ноль
    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])
        return d
