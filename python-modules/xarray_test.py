import wav_test
import xarray

def to_xarr(data, channels, sample_rate, sample_size):
    arr = xarray.DataArray(
        data,
        coords={'sample': len(data)/sample_size, 'channel': ['mono']},
        dims=['sample','channel'],
        name='audio_amplitude',
        attrs={'samplerate': sample_rate, 'samplesize': sample_size, 'units': 'Amplitude' }
    )
    return(arr)

import sys
if __name__ == "__main__":
    wav = wav_test.import_wav(sys.argv[1])
    arr = to_xarr(wav[0], wav[1], wav[2], wav[3])
    print(dir(arr))
    print(arr.coords)
    print(arr.dims)
    print(arr.name)
    print(arr.attrs)