from detectors.visual.handDetect import HandWatcher
from detectors.visual.neckDetect import LineDetector
from detectors.audio.noteDetect import NoteDetector

neckDetect = LineDetector()
chordDetect = HandWatcher()

chordDetect.detectChordShapes(0) # Use 0 for iphone camera 

