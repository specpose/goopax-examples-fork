def test_generator_simd(filename):
    import import_wav
    from import_wav import wav_format
    wav_format = wav_format(filename)
    print(wav_format)

    import gpu_reshaper
    from gpu_reshaper import silence
    descriptor = silence.host_query(wav_format[0], wav_format[1], denominator=128)

    print("Dumping memoryview descriptor:")
    print(descriptor)

    import struct
    from struct import calcsize
    for gpx_env,gpu_set in descriptor:
        for gpu in gpu_set:
            print(gpu[1][3]*calcsize(gpu[1][0]))

def test_wav_simd(filename):
    import import_wav
    from import_wav import wav_format
    wav_format = wav_format(filename)

    import gpu_reshaper
    from gpu_reshaper import silence
    descriptor = silence.host_query(wav_format[0], wav_format[1])

    print("Dumping memoryview descriptor:")
    print(descriptor)

    import device_selector
    from device_selector import retrieve_shape
    selection = retrieve_shape(descriptor)
    print("Using:")
    print(selection)

    from import_wav import import_wav_aligned
    custom_buffer = import_wav_aligned(filename,wav_format,selection)
    print("Imported data from file "+filename+".")

    Custom_Callable = silence(selection[0],bytes(selection[1]))

    mv = memoryview(custom_buffer)
    result = Custom_Callable(mv)
    del Custom_Callable

    print("Exporting data to file "+filename+".")
    from import_wav import export_data
    export_data(filename,custom_buffer)

if __name__ == "__main__":
    test_wav_simd('../output.wav')