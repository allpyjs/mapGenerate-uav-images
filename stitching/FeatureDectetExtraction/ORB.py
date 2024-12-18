import cv2
import numpy as np

# Load the image
img = cv2.imread('campus1.png')

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a named window for display
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
keypoints_orb, descriptors = orb.detectAndCompute(img_gray, None)

# Draw keypoints on the image
output_img = cv2.drawKeypoints(img, keypoints_orb, None, (255, 0, 255))

# Display the output image
cv2.imshow("output", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
