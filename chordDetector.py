from data_capture.object_detector.handDetector import HandWatcher
# from data_capture.note_detector.detectFrequency import FrequencyDetector

# freqDetect = FrequencyDetector()
chordDetect = HandWatcher()
videoPath = "./data_capture/videos/major7.MOV" # video directory for saved vids, 0 for input to be live webcam feed

# input_path = './data_capture/videos/standByMeChordsTrimmed.mp4'
# output_path = '/tmp/chordDetector/audioCut.wav'
# new_audio = '/tmp/chordDetector/audioFile.wav'

# freqDetect.extractAudio(input_path, output_path)
# freqDetect.convertMono(output_path, new_audio)
# frequencies = freqDetect.getFreq(new_audio)


# Ismael to create and call on new method in HandWatcher to create list of chordshapes with their respective timestamps
chordDetect.detectChordShapes(videoPath)
chordShapes = chordDetect.getChordShapes()
print(chordShapes)

# Jorge to create method for corelating list and merging them into one list 
# Corelate frequency and chordShape list
# for freq in frequencies:
#     freq[1] = round(freq[1]*1000)
# print(frequencies)

# Ismael to take third list output and write it to the vidfile at the respective frames  


##Logic to match the chordshapes with their respective frequencies at their respective frames based on the timestamps

