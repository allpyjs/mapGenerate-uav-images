import os
import cv2
import numpy as np
import pyexiv2


def restoreOrientation(image, orientation):
    if orientation == 8:
        restored_image = rotate(image, -90)
    elif orientation == 6:
        restored_image = rotate(image, 90)
    elif orientation == 3:
        restored_image = rotate(image, 180)
    else:
        restored_image = image

    return restored_image

def rotate(image, angle):
    # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    height = image.shape[0]
    width = image.shape[1]
    center = (width/2, height/2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # compute the new bounding dimensions of the image
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # adjust the rotation matrix to take into account translation
    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]

    # perform the actual rotation and return the image
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def get_metadata(input_file, camera_conf):
    filename = os.path.splitext(input_file)[0]

    if os.path.exists(filename + '.txt'):
        array = []
        with open(filename + '.txt', "r") as file:
            for line in file:
                parts = line.strip().split()
                filename, *values = parts
                values = [float(v) if '.' in v else int(v) for v in values]
                array.append([filename] + values)
            array = array[0]
    img = pyexiv2.Image(input_file)

    exif = img.read_exif()
    xmp = img.read_xmp()

    try:
        focal_length = convert_string_to_float(exif['Exif.Photo.FocalLength']) / 1000    # unit: m
        orientation = int(exif['Exif.Image.Orientation'])
        maker = exif["Exif.Image.Make"]

        longitude = convert_dms_to_deg(exif["Exif.GPSInfo.GPSLongitude"])
        latitude = convert_dms_to_deg(exif["Exif.GPSInfo.GPSLatitude"])

        lon_ref = exif['Exif.GPSInfo.GPSLongitudeRef']
        lat_ref = exif['Exif.GPSInfo.GPSLatitudeRef']

        if lon_ref == "W":
            longitude = -longitude
        if lat_ref == "S":
            latitude = -latitude

        if exif["Exif.Image.Make"] == "DJI":
            altitude = float(xmp['Xmp.drone-dji.RelativeAltitude'])
            roll = float(xmp['Xmp.drone-dji.GimbalRollDegree'])
            pitch = float(xmp['Xmp.drone-dji.GimbalPitchDegree'])
            yaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
        elif exif["Exif.Image.Make"] == "samsung":
            altitude = convert_string_to_float(exif['Exif.GPSInfo.GPSAltitude'])
            roll = float(xmp['Xmp.DLS.Roll']) * 180 / np.pi
            pitch = float(xmp['Xmp.DLS.Pitch']) * 180 / np.pi
            yaw = float(xmp['Xmp.DLS.Yaw']) * 180 / np.pi
        else:
            altitude = 0
            roll = 0
            pitch = 0
            yaw = 0

        eo = np.array(array[1:], dtype=float)
        omega = eo[3]
        phi = eo[4]
        kappa = eo[5]

        opk = np.array([omega, phi, kappa])

        eo = np.array([longitude, latitude, altitude, roll, pitch, yaw])

        return focal_length, orientation, eo, maker, opk
    except Exception:
        focal_length = camera_conf['focalLength'] / 1000   # unit: m
        maker = camera_conf['maker']
        orientation = 0
        eo = np.array(array[1:], dtype=float)
        eo[3] = eo[3] * 180 / np.pi
        eo[4] = eo[4] * 180 / np.pi
        eo[5] = eo[5] * 180 / np.pi

        omega = eo[3]
        phi = eo[4]
        kappa = eo[5]

        opk = np.array([omega, phi, kappa])


        return focal_length, orientation, eo, maker, opk

def convert_dms_to_deg(dms):
    dms_split = dms.split(" ")
    d = convert_string_to_float(dms_split[0])
    m = convert_string_to_float(dms_split[1]) / 60
    s = convert_string_to_float(dms_split[2]) / 3600
    deg = d + m + s
    return deg

def convert_string_to_float(string):
    str_split = string.split('/')
    return int(str_split[0]) / int(str_split[1])    # unit: mm
