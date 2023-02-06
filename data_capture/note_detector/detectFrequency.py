# Read in a WAV and find the freq's
import pyaudio
import wave
import numpy as np

from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

class FrequencyDetector:

    # def __init__(self):   
    #     self.vid = vid

    # def __str__(self, vid):
    #     return f"{self.vid}"

    def extractAudio(self, input_file, output_file):
        video = VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(output_file, fps=44100, nbytes=2, codec="pcm_s16le")

    def convertMono(self, input_audio, output_audio):
        sound = AudioSegment.from_wav(input_audio)
        sound = sound.set_channels(1)
        sound.export(output_audio, format="wav")
        
    # Try a BST(Binary Search Tree) for faster search
    def getNote(self, freq):

        notes = [
            ['C0', 16.35, []],
            ['C#0/Db0', 17.32, []],
            ['D0', 18.35, []],
            ['D#0/Eb0', 19.45, []],
            ['E0', 20.60, []],
            ['F0', 21.83, []],
            ['F#0/Gb0', 23.12, []],
            ['G0', 24.50, []],
            ['G#0/Ab0', 25.96, []],
            ['A0', 27.50, []],
            ['A#0/Bb0', 29.14, []],
            ['B0', 30.87, []],
            ['C1', 32.70, []],
            ['C#1/Db1', 34.65, []],
            ['D1', 36.71, []],
            ['D#1/Eb1', 38.89, []],
            ['E1', 41.20, []],
            ['F1', 43.65, []],
            ['F#1/Gb1', 46.25, []],
            ['G1', 49.00, []],
            ['G#1/Ab1', 51.91, []],
            ['A1', 55.00, []],
            ['A#1/Bb1', 58.27, []],
            ['B1', 61.74, []],
            ['C2', 65.41, []],
            ['C#2/Db2', 69.30, []],
            ['D2', 73.42, []],
            ['D#2/Eb2', 77.78, []],
            ['E2', 82.41, []],
            ['F2', 87.31, []],
            ['F#2/Gb2', 92.50, []],
            ['G2', 98.00, []],
            ['G#2/Ab2', 103.83, []],
            ['A2', 110.00, []],
            ['A#2/Bb2', 116.54, []],
            ['B2', 123.47, []],
            ['C3', 130.81, []],
            ['C#3/Db3', 138.59, []],
            ['D3', 146.83, []],
            ['D#3/Eb3', 155.56, []],
            ['E3', 164.81, []],
            ['F3', 174.61, []],
            ['F#3/Gb3', 185.00, []],
            ['G3', 196.00, []],
            ['G#3/Ab3', 207.65, []],
            ['A3', 220.00, []],
            ['A#3/Bb3', 233.08, []],
            ['B3', 246.94, []],
            ['C4', 261.63, []],
            ['C#4/Db4', 277.18, []],
            ['D4', 293.66, []],
            ['D#4/Eb4', 311.13, []],
            ['E4', 329.63, []],
            ['F4', 349.23, []],
            ['F#4/Gb4', 369.99, []],
            ['G4', 392.00, []],
            ['G#4/Ab4', 415.30, []],
            ['A4', 440.00, []],
            ['A#4/Bb4', 466.16, []],
            ['B4', 493.88, []],
            ['C5', 523.25, []],
            ['C#5/Db5', 554.37, []],
            ['D5', 587.33, []],
            ['D#5/Eb5', 622.25, []],
            ['E5', 659.25, []],
            ['F5', 698.46, []],
            ['F#5/Gb5', 739.99, []],
            ['G5', 783.99, []],
            ['G#5/Ab5', 830.61, []],
            ['A5', 880.00, []],
            ['A#5/Bb5', 932.33, []],
            ['B5', 987.77, []],
            ['C6', 1046.50, []],
            ['C#6/Db6', 1108.73, []],
            ['D6', 1174.66, []],
            ['D#6/Eb6', 1244.51	, []],
            ['E6', 1318.51, []],
            ['F6', 1396.91, []],
            ['F#6/Gb6', 1479.98, []],
            ['G6', 1567.98, []],
            ['G#6/Ab6', 1661.22, []],
            ['A6', 1760.00	, []],
            ['A#6/Bb6', 1864.66, []],
            ['B6', 1975.53	, []],
            ['C7', 2093.00, []],
            ['C#7/Db7', 2217.46, []],
            ['D7', 2349.32, []],
            ['D#7/Eb7', 2489.02, []],
            ['E7', 2637.02, []],
            ['F7', 2793.83, []],
            ['F#7/Gb7 ', 2959.96, []],
            ['G7', 3135.96, []],
            ['G#7/Ab7', 3322.44, []],
            ['A7', 3520.00, []],
            ['A#7/Bb7', 3729.31, []],
            ['B7', 3951.07, []],
            ['C8', 4186.01, []],
            ['C#8/Db8', 4434.92, []],
            ['D8', 4698.63, []],
            ['D#8/Eb8', 4978.03, []],
            ['E8', 5274.04, []],
            ['F8', 5587.65, []],
            ['F#8/Gb8', 5919.91, []],
            ['G8', 6271.93, []],
            ['G#8/Ab8', 6644.88	, []],
            ['A8', 7040.00, []],
            ['A#8/Bb8', 7458.62, []],
            ['B8', 7902.13, []],
        ]
        
        note = 'C0'
        # Iterate through each note's frequency
        for x in range(108):
            if(freq <= notes[x][1]):
            
                diff1 = notes[x][1] - freq
                diff2 = freq - notes[x-1][1]
                if(diff1 <= diff2):
                    note = notes[x][0]
                else :
                    note = notes[x-1][0]
                break
        return note 
    
    def getFreq(self, input_audio):

        freq_list = []
        note_list = []

        chunk = 2048

        # open up a wave
        wf = wave.open(input_audio, 'rb')
        # Returns sample width in bytes
        swidth = wf.getsampwidth()
        # Returns sampling frequency
        RATE = wf.getframerate()
        # Returns the number of frames
        frames = wf.getnframes()
        # Calculate the duration of the song/audio file
        duration = 0
        duration = frames/float(RATE)
        print("Duration: ", format(duration, '.4f'), " seconds.", "\n")

        # use a Blackman window
        window = np.blackman(chunk)
        # open stream
        p = pyaudio.PyAudio()
        stream = p.open(format =
                        p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = RATE,
                        output = True)
        print("Number of channels: ", wf.getnchannels())

        # read some data
        data = wf.readframes(chunk)

        # initialize count
        count = 0

        # play stream and find the frequency of each chunk
        while len(data) == chunk*swidth:
            count = count + 1
            # Print current position of file pointer
            position = wf.tell()
            #print("Position: ", position)

            # Print timestamps
            timestamp = position/float(RATE)
            #print("Timestamp: ", format(timestamp, '.4f'))

            # write data out to the audio stream
            stream.write(data)
            # unpack the data and times by the hamming window
            indata = np.array(wave.struct.unpack("%dh"%(len(data)/swidth),\
                                                data))*window
            # Take the fft and square each value
            fftData=abs(np.fft.rfft(indata))**2
            # find the maximum
            which = fftData[1:].argmax() + 1
            # use quadratic interpolation around the max
            # updates every 50ms
            if which != len(fftData)-1:
                y0,y1,y2 = np.log(fftData[which-1:which+2:])
                x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
                # find the frequency and output it
                thefreq = (which+x1)*RATE/chunk
                #print("The freq is %f Hz." %(thefreq))
            else:
                thefreq = which*RATE/chunk
                #print("The freq is %f Hz." %(thefreq))

            theNote = self.getNote(thefreq)

            # # Average frequencies after x amount of seconds
            # sumfreq = sumfreq + thefreq
            # if(count == 5):
            #     avgfreq = sumfreq/count
            #     count = 0
            #     print("Note is: ", findNote(avgfreq))
            #print("Note is: ", self.getNote(thefreq), "\n")#findNote(thefreq))

            # read some more data
            data = wf.readframes(chunk)

            note_list.append([theNote, timestamp])

        if data:
            stream.write(data)
        stream.close()
        p.terminate()

        return note_list
    