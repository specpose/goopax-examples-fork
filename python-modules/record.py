#!/bin/python
"""PyAudio Example: Record a few seconds of audio and save to a wave file."""

import wave
import sys

import pyaudio

CHUNK = 128
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 18.816 #rx460
#RECORD_SECONDS = 0.96 #hd530
#RECORD_SECONDS = 0.008 #corei7 would be continued fraction, so make 3x

with wave.open('output.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    print('Recording...')
    for _ in range(0, int(RATE // CHUNK * RECORD_SECONDS)):
        wf.writeframes(stream.read(CHUNK))
    print('Done')

    stream.close()
    p.terminate()
