import cv2
import numpy as np
img = cv2.imread('campus1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)


# this is in opencv-contrib-python
surf = cv2.xfeatures2d.SURF_create()

keypoints = surf.detect(img_gray, None)

cv2.imshow("output", cv2.drawKeypoints(img, keypoints, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()