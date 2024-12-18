import cv2
import numpy as np
# Load the image
img = cv2.imread('campus1.png')

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

sift = cv2.SIFT_create()

keypoints = sift.detect(img_gray, None)

cv2.imshow("output", cv2.drawKeypoints(img, keypoints, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()