import cv2
import os
import re

guitar_vids= "./data_capture/videos"
captured_frames="./data_capture/captured_frames"

for video in os.listdir(guitar_vids):
    video=video[0:video.find('.')]
    print(video)
    vidcap= cv2.VideoCapture(f"{guitar_vids}/{video}.mp4")
    success, image = vidcap.read()
    count = 0 
    while success:
        cv2.imwrite(f"{captured_frames}/{video}frame{count}.jpg", image)
        success, image = vidcap.read()
        print(str(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
        print(f'Read frame {count}: ', success)
        count += 1