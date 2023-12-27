#Intrinsics Interview Question: Coded by Ismael Banihamed :)
import numpy as np
import os
import cv2

##NOTES 
# edges: Output of the edge detector.
# lines: A vector to store the coordinates of the start and end of the line.
# rho: The resolution parameter \rho in pixels.
# theta: The resolution of the parameter \theta in radians.
# threshold: The minimum number of intersecting points to detect a line

class LineDetector():
    def __init__(self) -> None:
        self.imgOutputDir = "./data_capture/outputs"

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
    
    def horizontalLineDetect(self, image):
        kernel = np.ones((5,5),np.uint8)
        #Gray scale image
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)

        # Get rid of artifacts
        img = cv2.threshold(img, 250, 300, cv2.THRESH_BINARY)[1]

        # Create structuring elements
        horizontal_size = 10
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

        # Morphological opening 
        horizontalLines = cv2.morphologyEx(img, cv2.MORPH_OPEN, verticalStructure)
        # Dilate for further bolding of vertical lines 
        horizontalLines = cv2.morphologyEx(horizontalLines, cv2.MORPH_DILATE, kernel)

        return horizontalLines

    def sharpenImage(self, image):
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        image_sharp = cv2.filter2D(src=image, ddepth=-5, kernel=kernel)
        return image_sharp
    
    
    def run(self, imagePath, imageName):
        print(f"Detecting Lines for {imageName}")
        img = cv2.imread(imagePath)

        #Sharpen the image to have more contrast on borders 
        sharpenedImage = self.sharpenImage(img)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_sharpened.png',sharpenedImage)

        #get all edges drawn out using canny function
        grayImg = cv2.cvtColor(sharpenedImage, cv2.COLOR_RGB2GRAY)
        cannyImage = cv2.Canny(grayImg,100,150,apertureSize = 3)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_cannyEdges.png',cannyImage)

        # #get only the horizontal lines from image
        # mask = self.horizontalLineDetect(f'{self.imgOutputDir}/{imageName}_cannyEdges.png')
        # cv2.imwrite(f'{self.imgOutputDir}/{imageName}_horizontalEdges.png',mask)

        # #get the hough lines drawn out onto image
        lines = cv2.HoughLinesP(image=cannyImage,rho=1,theta=np.pi/180, threshold=60,lines=np.array([]), minLineLength=300,maxLineGap=80)
        height, width, color = img.shape
        minY = float('inf')
        maxY = 0
        for line in lines:
            print(line)
            for x1,y1,x2,y2 in line:
                yTotal = y1+y2
                y = y2 - y1
                x = x2 - x1
                slope = y/x
                if y != 0 and x != 0 and slope < 0 and slope > -1: 
                    if y2 < minY:
                        minY = y2
                        print(f'new min:{y2}')
                        topNeckLine = [x1,y1,x2,y2]
                    if y1 > maxY:
                        maxY = y1
                        bottomNeckLine = [x1,y1,x2,y2]
                    cv2.line(img, (x1,y1),(x2,y2), (255,0,0),1, cv2.LINE_AA)
        # Draw line for the top of the guitar neck
        cv2.line(img, (0,topNeckLine[1]),(width,topNeckLine[3]), (0,0,255),2, cv2.LINE_AA)
        # Draw line for the bottom of the guitar neck
        cv2.line(img, (0,bottomNeckLine[1]),(width,bottomNeckLine[3]), (0,0,255),2, cv2.LINE_AA)
        cropped_image = img[topNeckLine[3]:bottomNeckLine[1], 0:width]

        # Display the cropped image
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_cropped.png', cropped_image)
        cv2.imwrite(f'{self.imgOutputDir}/{imageName}_houghLines.png',img)
        


if __name__ == "__main__":
    inputsDir = "./data_capture/testImages/"
    Detector = LineDetector()

    for imageFile in os.listdir(inputsDir):
        imgPath = os.path.join(inputsDir, imageFile)
        print(imgPath)
        imgName = imageFile[:len(imageFile)-4] #discarding the .png from end of imagename
        Detector.run(imgPath, imgName)





    