def test_generator_simd(filename):
    from gpu_reshaper import silence
    descriptor = silence.host_query('h', denominator=128)

    print("Dumping memoryview descriptor:")
    print(descriptor)

    import device_selector
    from device_selector import retrieve_shape
    selection = retrieve_shape(descriptor)
    print("Using:")
    print(selection)

    from test_util import shape_to_len
    from struct import calcsize
    print("Generating minimal Input buffer size "+str(shape_to_len(selection[2]['shape'])*calcsize(selection[2]['format']))+" bytes...")
    from import_wav import generate_data_aligned
    custom_buffer = generate_data_aligned(selection)

    Custom_Callable = silence(selection[0],bytes(selection[1]))

    mv = memoryview(custom_buffer)
    result = Custom_Callable(mv)
    del Custom_Callable

    print("Exporting data to file "+filename+".")
    from import_wav import export_data
    export_data(filename,custom_buffer)

def test_wav_simd(filename):
    from import_wav import pcm_integer
    wav_format = pcm_integer(filename)
    print(wav_format)

    from gpu_reshaper import silence
    descriptor = silence.host_query(wav_format[0], length=wav_format[1])

    print("Dumping memoryview descriptor:")
    print(descriptor)

    selection = descriptor[1]
    print("Using:")
    print(selection)

    from import_wav import import_wav_aligned
    custom_buffer = import_wav_aligned(filename,selection)
    print("Imported data from file "+filename+".")

    Custom_Callable = silence(selection[0],bytes(selection[1]))

    mv = memoryview(custom_buffer)
    result = Custom_Callable(mv)
    del Custom_Callable

    print("Exporting data to file "+filename+".")
    from import_wav import export_wav
    export_wav(filename,custom_buffer)

if __name__ == "__main__":
    import sys
    from sys import path
    path.append('../python-modules')
    test_wav_simd('./output.wav')