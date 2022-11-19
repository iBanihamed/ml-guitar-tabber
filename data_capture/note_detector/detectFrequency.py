# Read in a WAV and find the freq's
import pyaudio
import wave
import numpy as np

chunk = 2048

# open up a wave
wf = wave.open('C:\\Users\\Jorge\\Documents\\GuitarProject\\ml-guitar-tabber\\data_capture\\note_detector\\StandByMe.wav', 'rb')
# Returns sample width in bytes
swidth = wf.getsampwidth()
print("Sample width = ", swidth)
# Returns sampling frequency
RATE = wf.getframerate()
# use a Blackman window
window = np.blackman(chunk)
# open stream
p = pyaudio.PyAudio()
stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)
# read some data
data = wf.readframes(chunk)
# play stream and find the frequency of each chunk
print("Length of data ", len(data))
while len(data) == chunk*swidth:
    print("enter while")
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
    if which != len(fftData)-1:
        y0,y1,y2 = np.log(fftData[which-1:which+2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        thefreq = (which+x1)*RATE/chunk
        print("The freq is %f Hz." %(thefreq))
        print("yes")
    else:
        thefreq = which*RATE/chunk
        print("The freq is %f Hz." %(thefreq))
        print("no")
    # read some more data
    data = wf.readframes(chunk)
if data:
    stream.write(data)
stream.close()
p.terminate()