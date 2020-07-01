import cv2

image = cv2.imread(r"D:\sher1.jpg")
resized = cv2.resize(image, (8, 8))
cv2.imshow("resized", resized)
cv2.waitKey(0)