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

def import_data(filename):
    header = import_header(filename)
    f = open(filename, 'r')
    if (header["audio_fmt"][0] != 1):
        raise Exception("Not PCM Integer")
    dt = numpy.dtype('i'+str(int(header["bits_per_sample"][0]/8)))
    #print(dt.isbuiltin)
    #print(dt.char)
    result = numpy.fromfile(f, dtype=dt, count=header["data_size"][0], sep='', offset=44)
    f.close()
    return result

import sys
if __name__ == "__main__":
    header = import_header(sys.argv[1])
    print(header)
    arr = import_data(sys.argv[1])
    print(arr[1151999])
    print(arr.itemsize)