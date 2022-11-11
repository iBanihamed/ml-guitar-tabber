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
        self.model_dir = "/Users/vn53q3k/OneDriveWalmartInc/MLPython/dl-test-automation/trained_models"
        self.testImagesDir = "/tmp/TestImages"
        self.failedImgDir = "/tmp/FailedImages"
        self.successImgDir = "/tmp/SuccessImages"
        self.reportDir = "/tmp/report.html"
        self.model = self.model_dir + "/model_ta.pt"
        self.imageType = ["home", "item", "search", "cart"]
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
            predClass = "Failure"
        elif prediction.item() == 1:
            predClass = "CartPage"
        elif prediction.item() == 2:
            predClass = "HomePage"
        elif prediction.item() == 3:
            predClass = "ItemPage"
        else:
            predClass = "SearchPage"
        return predClass

    def checkClass(self, image):
        for term in self.imageType:
            if term in image:
                return term.capitalize() + "Page"

    def test(self):
        model = Net(5)
        model.load_state_dict(torch.load(self.model, map_location='cpu'))
        results = {}
        if not os.path.exists(self.failedImgDir):
            os.makedirs(self.failedImgDir)
        if not os.path.exists(self.successImgDir):
            os.makedirs(self.successImgDir)
        total = 0
        failures = 0
        correct = 0
        for imageClass in os.listdir(self.testImagesDir):
            classDir = os.path.join(self.testImagesDir, imageClass)
            images = os.listdir(classDir)
            progressBar = Bar(imageClass, max=len(images))
            print(f"Calculating for {imageClass} class")
            results[imageClass] = {f"{term.capitalize()}Page": "NO DATA" for term in self.imageType}
            # first index represents pages that are calculated as fine,
            # second index represents pages that are calculated as failed
            for image in images:
                imgPath = os.path.join(classDir, image)
                img = self.convertImage(imgPath)
                prediction = self.predict(model, img)
                predictionClass = self.classifyPrediction(prediction)
                progressBar.message = f"Image: {image} |||| Model calculates it as: {predictionClass}"
                print("")
                classType = self.checkClass(image)
                if prediction.item() == 0:
                    shutil.copy2(imgPath, self.failedImgDir)
                    failures += 1
                    results[imageClass][classType] = "FAIL"
                else:
                    shutil.copy2(imgPath, self.successImgDir)
                    correct += 1
                    results[imageClass][classType] = "PASS"
                total += 1
                progressBar.next()
            progressBar.finish()
        print(f"Model calculated {correct/total * 100}% of pages loaded correctly for all classes")
        print(
            f"Model has found {failures} failures and has stored them in the following directory: {self.failedImgDir}"
        )
        if failures > 0:
            shutil.make_archive(self.failedImgDir, "zip", self.failedImgDir)
        return results

    def generateReport(self, resultDic):
        df = pandas.DataFrame.from_dict(resultDic, orient="index")
        result = df.to_html()
        textFile = open(self.reportDir, "w")
        textFile.write(result)
        print(df)


async def main():
    modelTester = OnnxTester()
    results = modelTester.test()
    modelTester.generateReport(results)


asyncio.run(main())
