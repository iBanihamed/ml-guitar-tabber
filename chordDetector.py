from data_capture.object_detector.handDetector import HandWatcher
from data_capture.object_detector.neckDetect import LineDetector
# from data_capture.note_detector.detectFrequency import FrequencyDetector

# freqDetect = FrequencyDetector()
neckDetect = LineDetector()
chordDetect = HandWatcher()

chordDetect.detectChordShapes(0) # Use 0 for iphone camera 

