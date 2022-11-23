from pydub import AudioSegment

sound = AudioSegment.from_wav("C:\\Users\\Jorge\\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\note_detector\\StandByMeCut.wav")
sound = sound.set_channels(1)
sound.export("C:\\Users\\Jorge\\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\note_detector\\file.wav", format="wav")