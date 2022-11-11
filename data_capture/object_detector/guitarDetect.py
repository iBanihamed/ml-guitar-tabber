#Intrinsics Interview Question: Coded by Ismael Banihamed :)
import numpy as np
import os
import cv2
import mediapipe as mp

from google.protobuf.json_format import MessageToDict

##NOTES 
# edges: Output of the edge detector.
# lines: A vector to store the coordinates of the start and end of the line.
# rho: The resolution parameter \rho in pixels.
# theta: The resolution of the parameter \theta in radians.
# threshold: The minimum number of intersecting points to detect a line

class LineDetector():
    def __init__(self) -> None:
        self.imgOutputDir = "/Users/vn53q3k/OneDriveWalmartInc/MLPython/guitarTabber/data_capture/transformed_images"
        pass

    def vertLineDetect(self, image):
        kernel = np.ones((5,5),np.uint8)
        #Gray scale image
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)

        # Get rid of artifacts
        img = cv2.threshold(img, 250, 300, cv2.THRESH_BINARY)[1]

        # Create structuring elements
        vertical_size = 10
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        # Morphological opening 
        vertLines = cv2.morphologyEx(img, cv2.MORPH_OPEN, verticalStructure)
        # Dilate for further bolding of vertical lines 
        vertLines = cv2.morphologyEx(vertLines, cv2.MORPH_DILATE, kernel)

        return vertLines

    def sharpenImage(self, image):
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        image_sharp = cv2.filter2D(src=image, ddepth=-5, kernel=kernel)
        return image_sharp
    
    def run(self, imagePath, imageName):
                # Initializing the Model
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=1)
        print(f"Detecting Lines for {imageName}")
        img = cv2.imread(imagePath)

        #Sharpen the image to have more contrast on borders 
        sharpenedImage = self.sharpenImage(img)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_sharpened.png',sharpenedImage)

        #get all edges drawn out using canny function
        grayImg = cv2.cvtColor(sharpenedImage, cv2.COLOR_RGB2GRAY)
        cannyImage = cv2.Canny(img,10,50,apertureSize =5)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_cannyEdges.png',cannyImage)

        #get only the vertical lines from image
        mask = self.vertLineDetect(f'{self.imgOutputDir}/{imageName}_cannyEdges.png')
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_vertEdges.png',mask)

        #get the hough lines drawn out onto image
        lines = cv2.HoughLinesP(image=mask,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=100,maxLineGap=80)

        verticalLineThreshold = 20
        leftBorderMin = 380
        leftBorderMax = 410
        rightBorderMin = 1050
        rightBorderMax = 1100
        xcordsLeft = []
        xcordsRight = []
        a,b,c = lines.shape
        for i in range(a):
            #conditional check to ensure coordinates are that of a line that is vertical enough before drawing onto image
            if abs(lines[i][0][0] - lines[i][0][2]) < verticalLineThreshold:
                if lines[i][0][0] > leftBorderMin and lines[i][0][0] < leftBorderMax:
                    #draw hough lines can comment out to negate extra lines
                    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 150, 0), 3, cv2.LINE_AA)
                    cv2.imwrite(f'{self.imgOutputDir}/{imageName}_houghLines.png',img)
                    #save coordinates for avergaing out for later use to draw one solid line for border
                    xcordsLeft.append(lines[i][0][0])
                    xcordsLeft.append(lines[i][0][2])
                elif lines[i][0][0] > rightBorderMin and lines[i][0][0] < rightBorderMax:
                    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 150, 0), 3, cv2.LINE_AA)
                    cv2.imwrite(f'{self.imgOutputDir}/{imageName}_houghLines.png',img)
                    xcordsRight.append(lines[i][0][0])
                    xcordsRight.append(lines[i][0][2])

        #Calculate and draw the average vertical linear line for left and right side of object in image 
        xcordRightAverage = sum(xcordsRight)/len(xcordsRight)
        xcordLeftAverage = sum(xcordsLeft)/len(xcordsLeft)
        height, width, color = img.shape
        cv2.line(img, (int(xcordLeftAverage), height), (int(xcordLeftAverage), 0), (0, 0, 150), 3, cv2.LINE_AA)
        cv2.line(img, (int(xcordRightAverage), height), (int(xcordRightAverage), 0), (0, 0, 150), 3, cv2.LINE_AA)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_houghLines.png',img)
        


if __name__ == "__main__":
    inputsDir = "/Users/vn53q3k/OneDriveWalmartInc/MLPython/guitarTabber/data_capture/transformed_images/"
    Detector = LineDetector()

    for imageFile in os.listdir(inputsDir):
        imgPath = os.path.join(inputsDir, imageFile)
        imgName = imageFile[:len(imageFile)-4] #discarding the .png from end of imagename
        Detector.run(imgPath, imgName)





    