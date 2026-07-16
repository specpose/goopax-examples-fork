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
    print(selection)

    from import_wav import import_wav_aligned
    custom_buffer = import_wav_aligned(filename,wav_format,selection)
    print("Imported data from file "+filename+".")

    arg1 = selection[0]
    print(type(arg1))
    print(arg1)
    arg2 = bytes(selection[1])
    print(type(arg2))
    print(arg2)
    print("uuid length is "+str(len(arg2)))
    Custom_Callable = silence(arg1,arg2)

    mv = memoryview(custom_buffer)
    result = Custom_Callable(mv)
    del Custom_Callable

    print("Exporting data to file "+filename+".")
    from import_wav import export_data
    export_data(filename,custom_buffer)

if __name__ == "__main__":
    test_wav_simd('../output.wav')