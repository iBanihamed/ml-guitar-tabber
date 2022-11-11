from __future__ import division
import json
import asyncio
from operator import itemgetter
import pandas
import os
import shutil
from progress.bar import Bar
import numpy as np
import cv2
import onnxruntime


from model import Net
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageEnhance

from env import settings

# This program takes a set of images and categorizes the failures into a directory
# reports on number of images that failed as well as what type it predicted it as for each image

config = settings()

class OnnxTester:
    def __init__(self) -> None:
        self.img_dir = "/tmp/ImgScreenshots"
        self.model_dir = "./trained_models"
        self.testImagesDir = "/tmp/TestImages"
        self.failedImgDir = "/tmp/FailedImages"
        self.successImgDir = "/tmp/SuccessImages"
        self.reportDir = "/tmp/report.html"
        self.model = self.model_dir + "/model_guitar_tabber.pt"
        self.imageType = ["A_chord", "D_chord", "Dmaj7_chord", "G_chord"]
        self.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...

    def convertImage(self, imgPath):
        img = Image.open(imgPath).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img

    def predict(self, model, img):
        output = model(img)
        print(output)
        pred = torch.argmax(output, 1)
        return pred

    def classifyPrediction(self, prediction):
        if prediction.item() == 0:
            predClass = "A_chord"
        elif prediction.item() == 1:
            predClass = "D_chord"
        elif prediction.item() == 2:
            predClass = "Dmaj7_chord"
        elif prediction.item() == 3:
            predClass = "G_chord"
        else:
            predClass = "NODATA"
        return predClass

    def checkClass(self, image):
        for term in self.imageType:
            if term in image:
                return term.capitalize() + "Page"

    def test(self):
        model = Net(5)
        model.load_state_dict(torch.load(self.model, map_location='cpu'))
        for imageClass in os.listdir(self.testImagesDir):
            classDir = os.path.join(self.testImagesDir, imageClass)
            images = os.listdir(classDir)
            progressBar = Bar(imageClass, max=len(images))
            print(f"Calculating for {imageClass} class")
            for image in images:
                imgPath = os.path.join(classDir, image)
                img = self.convertImage(imgPath)
                prediction = self.predict(model, img)
                predictionClass = self.classifyPrediction(prediction)
                progressBar.message = f"Image: {image} |||| Model calculates it as: {predictionClass}"
                progressBar.next()
            progressBar.finish()

async def main():
    modelTester = OnnxTester()
    results = modelTester.test()


asyncio.run(main())
