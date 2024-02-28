from __future__ import division
import asyncio
import os
# from progress.bar import Bar
import numpy as np

from model.model import Net
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageEnhance


class ModelInferencer:
    def __init__(self) -> None:
        self.img_dir = "/tmp/ImgScreenshots"
        self.model_dir = "./model/trained_models/model_guitar_tabber.pt"
        self.guitar_vids = './data_capture/videos'
        self.model = Net(3)
        self.model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))
        self.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...
        self.chords = {
            0: 'major', 
            1: 'major7',
            2: 'minor'
            # 3: 'thumb'
        }

    def convertImage(self, imgPath):
        img = Image.open(imgPath).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img

    def predict(self, img):
        output = self.model(img)
        pred = torch.argmax(output, 1)
        return pred

    def classifyPrediction(self, prediction):
        return self.chords[prediction.item()]
