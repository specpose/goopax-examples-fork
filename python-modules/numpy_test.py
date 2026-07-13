import numpy

wav_header_dtype = numpy.dtype([
    ("chunk_id", (bytes, 4)), # flexible-sized scalar type, item size 4
    ("chunk_size", "<u4"),    # little-endian unsigned 32-bit integer
    ("format", "S4"),         # 4-byte string
    ("fmt_id", "S4"),
    ("fmt_size", "<u4"),
    ("audio_fmt", "<u2"),     #
    ("num_channels", "<u2"),  # .. more of the same ...
    ("sample_rate", "<u4"),   #
    ("byte_rate", "<u4"),
    ("block_align", "<u2"),
    ("bits_per_sample", "<u2"),
    ("data_id", ("S1", (2, 2))), # sub-array, just for fun!
    ("data_size", "u4"),
    #
    # the sound data itself cannot be represented here:
    # it does not have a fixed size
   ])

def import_header(filename):
    f = open(filename, 'r')
    result = numpy.fromfile(f, dtype=wav_header_dtype, count=1)
    f.close()
    return result

def wav_format(filename):
    header = import_header(filename)
    f = open(filename, 'r')
    if (header["audio_fmt"][0] != 1):
        raise Exception("Not PCM Integer")
    f.close()
    return ('h',int(header["data_size"][0]))

def import_wav(filename, wav_format):
    header = import_header(filename)
    f = open(filename, 'r')
    if (header["audio_fmt"][0] == 1 and wav_format[0] != 'h'):
        raise Exception("Only short int is accepted")
    dt = numpy.dtype('i'+str(int(header["bits_per_sample"][0]/8)))
    result = numpy.fromfile(f, dtype=dt, count=header["data_size"][0], sep='', offset=44)
    f.close()
    return result

def import_wav_aligned(filename, wav_format, shape):
    header = import_header(filename)
    f = open(filename, 'rb')
    if (header["audio_fmt"][0] == 1 and wav_format[0] != 'h'):
        raise Exception("Only short int is accepted")
    import gpu_buffer
    from gpu_buffer import CustomBuffer
    buffy = CustomBuffer(wav_format[0], shape[2], shape[3])
    bview = memoryview(buffy)
    f.read(44)
    if not (f.readinto(bview) == header["data_size"][0]):
        raise Exception("Could not read file into numpy array")
    #print(bview.tobytes())
    bview.release()
    f.close()
    return buffy

def export_data(filename, buffy):
    header = import_header(filename)
    f = open(filename, 'rb+')
    if (header["audio_fmt"][0] != 1):
        raise Exception("Not PCM Integer")
    f.seek(44)
    mview = memoryview(buffy)
    print(type(mview))
    print(mview)
    #print(mview.tobytes())
    f.write(mview.tobytes())
    mview.release()
    f.close()

import sys
if __name__ == "__main__":
    pass