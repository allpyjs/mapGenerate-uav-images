import cv2
import numpy as np

# Load the image
img = cv2.imread('campus1.png')

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners using cv2.goodFeaturesToTrack
corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 1)

# Convert corner coordinates to integers
corners = np.round(corners).astype(int)

# Create a named window for display
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

# Draw circles for each corner
for i in corners:
    x, y = i.ravel()  # Extract x and y from the corner
    cv2.circle(img_1, (x, y), 3, (255, 0, 255), -1)  # Draw the circle

# Display the result
cv2.imshow("output", img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
