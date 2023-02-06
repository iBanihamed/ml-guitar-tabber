from detectFrequency import FrequencyDetector

det = FrequencyDetector()

input_path = 'C:\\Users\\Jorge\\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\videos\\standByMeChordsTrimmed.mp4'
output_path = 'C:\\Users\\Jorge\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\note_detector\\audio_files\\StandByMeCut.wav'
new_audio = 'C:\\Users\\Jorge\\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\note_detector\\audio_files\\audioFile.wav'

det.extractAudio(input_path, output_path)
det.convertMono(output_path, new_audio)
print(det.getFreq(new_audio))
