import cv2
import numpy as np

"""
This file contains the soft criteria:
1. Count the blur score
2. Check the gaze direction

"""


def detect_blur(image):
    """
    :param image: The gray scale image to be analyzed
    :return: The variance of laplacian of the image
    """
    score = np.var(cv2.Laplacian(image, cv2.CV_8U))
    return score


def looking_center(gray, shape, side, debug=False):
    """
    :param gray: The gray scale image which contains a face
    :param shape: The coordinates of facial landmarks
    :param side: Which side of the eyes will be analyzed
    :param debug: Boolean value if some helpful information should be printed
    :return: Gaze score. If the score == -2, then the frame is not suitable because it is either:
    1. No pupil detected
    2. Look extremely up
    3. Look extremely left or right
    """
    if side == "left_eye":
        region = shape[36:42]
    else:
        region = shape[42:48]
    left_best, right_best = 0.8, 1.2
    left_good, right_good = 0.5, 1.9
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    margin = 2
    min_x = np.min(region[:, 0]) + margin
    max_x = np.max(region[:, 0]) - margin
    min_y = np.min(region[:, 1]) + margin
    max_y = np.max(region[:, 1]) - margin
    gray_eye = eye[min_y: max_y, min_x: max_x]
    # only eye are cropped
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    if threshold_eye is None:
        return -2
    # cv2_imshow(threshold_eye)
    height, width = threshold_eye.shape
    # left part and right part of the eye
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]

    # number of black pixels in left and right part of the eye
    left_side_black = height * int(width / 2) - cv2.countNonZero(left_side_threshold)
    right_side_black = height * int(width / 2) - cv2.countNonZero(right_side_threshold)

    # up and down part of the eye
    up_side_threshold = threshold_eye[0: int(height / 2), 0: width]
    down_side_threshold = threshold_eye[int(height / 2): height, 0: width]

    # number of black pixels in left and right part of the eye
    up_side_black = int(height / 2) * width - cv2.countNonZero(up_side_threshold)
    down_side_black = height * int(width / 2) - cv2.countNonZero(down_side_threshold)
    white = cv2.countNonZero(threshold_eye)
    black = height * width - white
    if (white == 0) or (down_side_black < 10) or (up_side_black < 10) or (black / white) < 0.4:
        if debug:
            print("Looking up or no pupil")
        return -2
    if right_side_black == 0 or left_side_black == 0:
        if debug:
            print("Looking extremely left or right")
        return -2
    gaze_ratio_horizontal = left_side_black / right_side_black
    if debug:
        print("Ratio", gaze_ratio_horizontal, "Black", black, "White", white)
    if int(left_best <= gaze_ratio_horizontal <= right_best):
        return 1
    elif int(left_good <= gaze_ratio_horizontal <= right_good):
        return 0.8
    else:
        return -1
