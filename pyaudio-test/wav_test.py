import wave

def import_wav(name):
    with wave.open(name, 'rb') as wf:
        data = wf.readframes(-1)
        print(type(data))
        print(len(data))
        channels = wf.getnchannels()
        print(channels)
        sample_rate = wf.getframerate()
        print(sample_rate)
        sample_size = wf.getsampwidth()
        print(sample_size)
        return (data, channels, sample_rate, sample_size)