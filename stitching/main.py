import sys
import cv2
import numpy as np
import imutils
from imutils import paths
import argparse


class Orthomosaic:
    def __init__(self, debug):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.no_raw_images = []
        self.temp_image = []
        self.final_image = []
        self.debug = debug

    def load_dataset(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-i", "--images", type=str, required=True,
                             help="path to input directory of images to stitch")
        self.ap.add_argument("-o", "--output", type=str, required=True,
                             help="path to the output image")
        self.args = vars(self.ap.parse_args())

        # grab the paths to the input images and initialize our images list
        if self.debug:
            print("[INFO] Importing Images...")
        self.imagePaths = sorted(list(paths.list_images(self.args["images"])))
        self.images = []
        for imagePath in self.imagePaths:
            self.image_temp = cv2.imread(imagePath)
            scale_percent = 50  # Adjust to resize input images to 50% of original size
            width = int(self.image_temp.shape[1] * scale_percent / 100)
            height = int(self.image_temp.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image to smaller dimensions
            self.image = cv2.resize(self.image_temp, dim)
            self.images.append(self.image)
        if self.debug:
            print("[INFO] Importing Complete")

    def mixer(self):
        self.no_raw_images = len(self.images)
        if self.debug:
            print(f"[INFO] {self.no_raw_images} Images have been loaded")
        for x in range(self.no_raw_images):
            if x == 0:
                self.temp_image = self.sticher(self.images[x], self.images[x+1])
            elif x < self.no_raw_images - 1:
                self.temp_image = self.sticher(self.temp_image, self.images[x+1])
            else:
                self.final_image = self.temp_image

        cv2.imshow("output", self.final_image)
        cv2.imwrite(self.args["output"], self.final_image)  # Save the result
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sticher(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        orb = cv2.ORB_create(nfeatures=1000)

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(self.image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.image2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        all_matches = []
        for m, n in matches:
            all_matches.append(m)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

        # Set minimum match condition
        MIN_MATCH_COUNT = 0

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            result = self.wrap_images(image2, image1, M)
            return result
        else:
            print("Error: Not enough matches")
            return None

    def wrap_images(self, image1, image2, H):
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]
        
        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        
        # When we have established a homography we need to warp perspective
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
        
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        # Print the values of x_min, y_min, x_max, and y_max for debugging
        print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

        # Clipping the dimensions to avoid exceeding OpenCV's limits (SHRT_MAX)
        max_size = 32767  # max value for SHRT_MAX
        x_max = min(x_max, max_size)
        y_max = min(y_max, max_size)

        # If the image is still too large, scale it down
        if x_max > max_size or y_max > max_size:
            scaling_factor = min(max_size / float(x_max), max_size / float(y_max))
            x_max = int(x_max * scaling_factor)
            y_max = int(y_max * scaling_factor)

        # Make sure the dimensions do not fall below a reasonable size
        x_max = max(x_max, 100)  # Enforce a minimum size for x_max
        y_max = max(y_max, 100)  # Enforce a minimum size for y_max

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], 
                                  [0, 1, translation_dist[1]], 
                                  [0, 0, 1]])

        output_img = cv2.warpPerspective(image2, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        output_img[translation_dist[1]:rows1 + translation_dist[1], 
                   translation_dist[0]:cols1 + translation_dist[0]] = image1

        return output_img


if __name__ == "__main__":
    tester = Orthomosaic(debug=True)
    tester.load_dataset()
    tester.mixer()
