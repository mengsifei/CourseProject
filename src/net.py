import torch.nn as nn
import dlib
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision import models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def load_efficientnet():
    efficientnet_v2_s = models.efficientnet_v2_s()
    efficientnet_v2_s.classifier[1] = nn.Linear(1280, 1, bias=True)
    checkpoint = torch.load('data/efficientnetsmall.pth', map_location='cpu')
    efficientnet_v2_s.load_state_dict(checkpoint['model_state_dict'])
    efficientnet_v2_s = efficientnet_v2_s.to(device)
    return efficientnet_v2_s


def predict_img(img, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = Image.fromarray(img)
    model.eval()
    with torch.no_grad():
        img = transform(img)
        # make dim 4D
        img = img.unsqueeze(0).to(device)
        output = model(img).cpu()[0][0]
        score = np.array(output)
        return score


