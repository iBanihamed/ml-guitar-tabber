from data_capture.object_detector.handDetector import HandWatcher
from data_capture.note_detector.detectFrequency import FrequencyDetector

freqDetect = FrequencyDetector()
chordDetect = HandWatcher()

videoPath = "./data_capture/videos/major7.MOV" # video directory for saved vids, 0 for input to be live webcam feed
# input_path = './data_capture/videos/standByMeChordsTrimmed.mp4'
output_path = './data_capture/note_detector/audio_files/audioCut.wav'
new_audio = './data_capture/note_detector/audio_files/audioFile.wav'

freqDetect.extractAudio(videoPath, output_path)
freqDetect.convertMono(output_path, new_audio)
frequencies = freqDetect.getFreq(new_audio)
#print(frequencies)

# Ismael to create and call on new method in HandWatcher to create list of chordshapes with their respective timestamps
chordDetect.detectChordShapes(videoPath)
chordShapes = chordDetect.getChordShapes() #list of chords
#print(chordShapes)

# setting length of chordList 
if(len(frequencies) < len(chordShapes)):
    length = len(frequencies)
else:
    length = len(chordShapes)

# Jorge to create method for corelating list and merging them into one list 
# Corelate frequency and chordShape list
chordList = []

for i in range(length):
    chordList.append(frequencies[i] + chordShapes[i])

print("Final List: ")
print(chordList)

# Ismael to take third list output and write it to the vidfile at the respective frames  


##Logic to match the chordshapes with their respective frequencies at their respective frames based on the timestamps

