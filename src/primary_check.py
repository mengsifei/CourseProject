import numpy as np


def norm(num1, num2):
    return np.sqrt(np.sum((np.array(num1) - np.array(num2)) ** 2))


def eye_aspect_ratio(side, shape):
    threshold = 0.26
    if side == "left_eye":
        region = shape[36:42]
    else:
        region = shape[42:48]
    p1, p2, p3, p4, p5, p6 = region[0], region[1], region[2], region[3], region[4], region[5]
    ear = (norm(p2, p6) + norm(p3, p5)) / (2.0 * norm(p1, p4))
    if ear < threshold:
        return True, 0
    return False, ear


def mouth_dist(shape):
    threshold = 0.09
    mouth = shape[60:68]
    mar = (norm(mouth[2], mouth[6]) + norm(mouth[1], mouth[7]) + norm(mouth[3], mouth[5])) / (2 * norm(mouth[0], mouth[4]))
    # mar = (norm(mouth[2], mouth[6]) + norm(mouth[1], mouth[7]) + norm(mouth[3], mouth[5])) / 3
    if mar == 0:
        return True, 2
    return mar >= threshold, mar