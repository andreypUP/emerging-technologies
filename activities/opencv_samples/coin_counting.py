# Here is the simple coin counter. You may improve this program.

import cv2
import numpy as np

image = cv2.imread("coins.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11,11), 0)
# cv2.imshow("Image", image)
edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("Edges", edged)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow('Coins', coins)
cv2.waitKey(0)