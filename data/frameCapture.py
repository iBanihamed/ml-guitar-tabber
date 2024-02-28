import cv2
import os
import re
from detectors.visual.neckDetect import LineDetector
from detectors.visual.handDetect import HandWatcher

guitar_vids= "./data_capture/videos"

for video in os.listdir(guitar_vids):
    # extract string before "." in file name
    video=video[0:video.find('.')]  

    # Create a path to save the frames to 
    framesPath = f"./data_capture/captured_frames/{video}"  
    if not os.path.exists(framesPath):
        os.mkdir(framesPath)

    # Open up the video for reading
    vidcap= cv2.VideoCapture(f"{guitar_vids}/{video}.mov")
    success, image = vidcap.read()
    count = 0 
    while success:
        # Write every successful captured frame to a file in framespath
        cv2.imwrite(f"{framesPath}/{video}frame{count}.jpg", LineDetector().run(HandWatcher().cropHand(image)))
        success, image = vidcap.read()
        print(str(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        print(f'Read frame {count}: ', success)
        count += 1