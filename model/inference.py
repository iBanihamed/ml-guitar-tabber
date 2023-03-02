from __future__ import division
import asyncio
import os
from progress.bar import Bar
import numpy as np

from model.model import Net
import torch
torch.manual_seed(0)
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageEnhance


class ModelInferencer:
    def __init__(self) -> None:
        self.img_dir = "/tmp/ImgScreenshots"
        self.model_dir = "./model/trained_models/model_guitar_tabber.pt"
        self.model = Net(4)
        self.model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))
        self.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...

    def convertImage(self, imgPath):
        img = Image.open(imgPath).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img

    def predict(self, img):
        output = self.model(img)
        #print(output)
        pred = torch.argmax(output, 1)
        return pred

    def classifyPrediction(self, prediction):
        if prediction.item() == 0:
            predClass = "major"
        elif prediction.item() == 1:
            predClass = "major7"
        elif prediction.item() == 2:
            predClass = "minor"
        elif prediction.item() == 3:
            predClass = "minor7"
        else:
            predClass = "NODATA"
        return predClass

    # def loadModel(self, modelSize, mapLocation='cpu'):
    #     self.model = Net(modelSize)
    #     self.mapLocation = mapLocation
    #     self.model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))

    # def test(self):
    #     model = Net(4)
    #     model.load_state_dict(torch.load(self.model, map_location='cpu'))
    #     for imageClass in os.listdir(self.testImagesDir):
    #         classDir = os.path.join(self.testImagesDir, imageClass)
    #         images = os.listdir(classDir)
    #         progressBar = Bar(imageClass, max=len(images))
    #         print(f"Calculating for {imageClass} class")
    #         for image in images:
    #             imgPath = os.path.join(classDir, image)
    #             img = self.convertImage(imgPath)
    #             prediction = self.predict(model, img)
    #             predictionClass = self.classifyPrediction(prediction)
    #             progressBar.message = f"Image: {image} |||| Model calculates it as: {predictionClass}"
    #             progressBar.next()
    #         progressBar.finish()

# async def main():
#     modelTester = ModelInferencer()
#     results = modelTester.test()


# asyncio.run(main())
