from rgbhist import RGBHist
from searcher import Searcher
import numpy as np
import argparse
import os
import pickle
import cv2

# Обработка команд в консоли с обработкой трех обязательных параметров:
# 1. Название папки-датасета
# 2. Файл с индексами
# 3. Путь к ищображению-образцу
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=True,
                help="Path to where we stored our index")
ap.add_argument("-q", "--query", required=True,
                help="Path to query image")
args = vars(ap.parse_args())
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", queryImage)
print("query: {}".format(args["query"]))
desc = RGBHist([8, 8, 8])
queryFeatures = desc.describe(queryImage)
index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)
results = searcher.search(queryFeatures)        # Выполняется поиск
# Вывод первых 10 результатов в формате "путь файла : хи-квадрат"
# Если изображения категории образца входят в первые 10 позиций
# Поиск считается успешным
for j in range(0, 10):
    (score, imageName) = results[j]
    path = os.path.join(args["dataset"], imageName)
    result = cv2.imread(path)
    print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
cv2.waitKey(0)
