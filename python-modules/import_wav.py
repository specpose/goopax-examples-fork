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

def pcm_integer(filename):
    header = import_header(filename)
    if (header["audio_fmt"][0] != 1):
        raise Exception("Not PCM Integer")
    from struct import calcsize
    return ('h',int(header["data_size"][0]/calcsize('h')))

def import_wav(filename, selection):
    header = import_header(filename)
    f = open(filename, 'r')
    if (selection[2]['format'] != 'h'):
        raise Exception("Only short int is accepted")
    dt = numpy.dtype('i'+str(int(header["bits_per_sample"][0]/8)))
    arr = numpy.fromfile(f, dtype=dt, count=header["data_size"][0], sep='', offset=44)
    f.close()
    return arr

def import_wav_aligned(filename, selection):
    header = import_header(filename)
    f = open(filename, 'rb')
    if (selection[2]['format'] != 'h'):
        raise Exception("Only short int is accepted")
    from gpu_buffer import CustomBuffer
    buffy = CustomBuffer(selection[2]['format'], selection[2]['shape'], selection[2]['alignment'])
    bview = memoryview(buffy)
    f.read(44)
    if not (f.readinto(bview) == header["data_size"][0]):
        raise Exception("Could not read file into buffy")
    bview.release()
    f.close()
    return buffy

def generate_data_aligned(selection):
    from gpu_buffer import CustomBuffer
    buffy = CustomBuffer(selection[2]['format'], selection[2]['shape'], selection[2]['alignment'])
    bview = memoryview(buffy)
    bview.release()
    return buffy

def export_wav(filename, buffy):
    header = import_header(filename)
    f = open(filename, 'rb+')
    if (header["audio_fmt"][0] != 1):
        raise Exception("Not PCM Integer")
    f.seek(44)
    mview = memoryview(buffy)
    f.write(mview.tobytes())
    mview.release()
    f.close()

def export_data(filename, buffy):
    f = open(filename, 'wb')
    mview = memoryview(buffy)
    f.write(mview.tobytes())
    mview.release()
    f.close()