import torch.nn as nn
import dlib
import torch
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision import models

class LightNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.init_bias()  # initialize bias

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def init_bias(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                layer.weight = nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_net():
    net = LightNet()
    net = net.to(device)
    checkpoint = torch.load('data/pretrained.pth', map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
    return net


def load_efficientnet():
    efficientnet_v2_s = models.efficientnet_v2_s()
    efficientnet_v2_s.classifier[1] = nn.Linear(1280, 1, bias=True)
    checkpoint = torch.load('data/efficientnetsmall.pth', map_location=torch.device('cpu'))
    efficientnet_v2_s.load_state_dict(checkpoint['model_state_dict'])
    efficientnet_v2_s = efficientnet_v2_s.to(device)
    return efficientnet_v2_s


# set global variables
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
landmark_dict = {"left_eye": (42, 48), "right_eye": (36, 48), "inner_mouth": (60, 68)}


def blur_score(image):
    score = np.var(cv2.Laplacian(image, cv2.CV_64F))
    return score


def norm(num1, num2):
    return np.sqrt(np.sum((np.array(num1) - np.array(num2)) ** 2))


def looking_center(gray, region, ear):
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    if threshold_eye is None:
        return True, ear
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    up_side_threshold = threshold_eye[0: int(height / 2), 0: width]
    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[int(height / 2): height, 0:width]
    down_side_white = cv2.countNonZero(down_side_threshold)
    if right_side_white == 0 or down_side_white == 0:
        return True, ear
    gaze_ratio_horizontal = left_side_white / right_side_white
    gaze_ratio_vertical = up_side_white / down_side_white
    return ((1 > gaze_ratio_horizontal) or (gaze_ratio_horizontal > 1.6)) or ((1 > gaze_ratio_vertical) or
                                                                              (gaze_ratio_vertical > 1.6)), ear


def eye_aspect_ratio(side, gray, shape, is_first):
    threshold = 0.26
    region = shape[landmark_dict[side][0]:landmark_dict[side][1]]
    p1, p2, p3, p4, p5, p6 = region[0], region[1], region[2], region[3], region[4], region[5]
    ear = (norm(p2, p6) + norm(p3, p5)) / (2.0 * norm(p1, p4))
    if ear < threshold:
        return True, 0
    # gaze
    if not is_first:
        return looking_center(gray, region, ear)
    return False, ear


def predict_img(img, model):
    img = Image.fromarray(img)
    model.eval()
    with torch.no_grad():
        img = transform(img)
        # make dim 4D
        img = img.unsqueeze(0).to(device)
        output = model(img).cpu()[0][0]
        score = np.array(output)
        return score


def mouth_dist(shape):
    threshold = 0.15
    mouth = shape[landmark_dict['inner_mouth'][0]: landmark_dict['inner_mouth'][1]]
    mar = (norm(mouth[2], mouth[6]) + norm(mouth[1], mouth[7]) + norm(mouth[3], mouth[5])) / (2 * norm(mouth[0], mouth[4]))
    return mar >= threshold, mar

