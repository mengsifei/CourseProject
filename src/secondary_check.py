import cv2
import numpy as np


def detect_blur(image):
    score = np.var(cv2.Laplacian(image, cv2.CV_64F))
    return score


def looking_center(gray, shape, side):
    if side == "left_eye":
        region = shape[36:42]
    else:
        region = shape[42:48]
    left_best, right_best = 0.75, 1.3
    left_good, right_good = 0.5, 2.5
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    margin = 1
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
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_black = height * int(width / 2) - cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_black = height * int(width / 2) - cv2.countNonZero(right_side_threshold)
    up_side_threshold = threshold_eye[0: int(height / 2), 0: width]
    up_side_black = int(height / 2) * width - cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
    down_side_black = height * int(width / 2) - cv2.countNonZero(down_side_threshold)
    # print("down_side_black", down_side_black, "up_side_black", up_side_black)
    white = cv2.countNonZero(threshold_eye)
    black = height * width - white
    if (white == 0) or (down_side_black < 10) or (up_side_black < 10):
        return -2
    if (black / white) < 0.4:
        return -2
    if right_side_black == 0 or left_side_black == 0:
        return -2
    gaze_ratio_horizontal = left_side_black / right_side_black
    # print(side, "gaze_ratio_horizontal", gaze_ratio_horizontal, "left", left, "right", right)
    if int(left_best <= gaze_ratio_horizontal <= right_best):
        return 1
    elif int(left_good <= gaze_ratio_horizontal <= right_good):
        return 0.5
    else:
        return -1


def head_pose(gray, shape):
    model_points = np.array([[0.0, 0.0, 0.0],  # Tip of the nose [30]
                        [0.0, -330.0, -65.0],  # Chin [8]
                        [-225.0, 170.0, -135.0],  # Left corner of the left eye  [45]
                        [225.0, 170.0, -135.0],  # Right corner of the right eye [36]
                        [-150.0, -150.0, -125.0],  # Left corner of the mouth [54]
                        [150.0, -150.0, -125.0]])  # Right corner of the mouth [48]
    width = gray.shape[1]
    center = (gray.shape[1] / 2, gray.shape[0] / 2)
    camera_matrix = np.array([[width, 0, center[0]],
                                    [0, width, center[1]],
                                    [0, 0, 1]], dtype="double")
    distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
    image_points = shape[[30, 8, 45, 36, 54, 48]]  # pick points corresponding to the model
    success, rotation_vec, translation_vec = \
        cv2.solvePnP(model_points.astype('float32'), image_points.astype('float32'), camera_matrix, distortion_coefficients)
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    angles[0, 0] = - angles[0, 0]
    return angles[1, 0], angles[0, 0], angles[2, 0]