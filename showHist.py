# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np

import cv2


def show_hist(img):
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
    plt.show()

image = cv2.imread(r"queries/yellow_cat.jpg")
cv2.imshow("image", image)
show_hist(image)
image = cv2.imread(r"images/yellow_train.jpg")
cv2.imshow("image", image)
show_hist(image)