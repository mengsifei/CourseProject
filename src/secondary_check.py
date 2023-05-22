import cv2
import numpy as np


def detect_blur(image):
    score = np.var(cv2.Laplacian(image, cv2.CV_8U))
    return score


def looking_center(gray, shape, side, debug=False):
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
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    if threshold_eye is None:
        return -2
    # cv2_imshow(threshold_eye)
    height, width = threshold_eye.shape
    left_threshold = threshold_eye[0: height, 0: int(width / 2)]
    right_threshold = threshold_eye[0: height, int(width / 2): width]
    left_white = cv2.countNonZero(threshold_eye[0: height, 0: int(width / 4)])
    left_black = height * int(width / 2) - cv2.countNonZero(left_threshold)
    right_white = cv2.countNonZero(threshold_eye[0: height, int(3 * width / 4): width])
    right_black = height * int(width / 2) - cv2.countNonZero(right_threshold)
    down_threshold = threshold_eye[int(4 * height / 5): height, 0: width]
    down_black = down_threshold.shape[0] * down_threshold.shape[1] - cv2.countNonZero(down_threshold)
    white = cv2.countNonZero(threshold_eye)
    black = height * width - white
    if debug:
        cv2_imshow(threshold_eye)
        print(side, "black", black, "white", white, "ratio", black / white, "left white", left_white, "right_white",
              right_white, "down_black", down_black)
    if (white == 0) or (down_black <= 10):
        if debug:
            print("Looking up")
        return -2

    if (black / white) < 0.4:
        if debug:
            print("no pupil")
        return -2
    if right_white < 3 or left_white < 3 or right_black == 0:
        if debug:
            print("looking extreme left or right")
        return -2
    gaze_ratio_horizontal = left_black / right_black
    # print(side, "gaze_ratio_horizontal", gaze_ratio_horizontal)
    if int(left_best <= gaze_ratio_horizontal <= right_best):
        return 1
    elif int(left_good <= gaze_ratio_horizontal <= right_good):
        return 0.5
    else:
        return -1
