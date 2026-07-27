def retrieve_shape(memoryview_descriptor):
    by_workgroupsize = sorted(memoryview_descriptor, key=lambda d: d[2]["shape"][1], reverse=True)
    by_register = sorted(by_workgroupsize, key=lambda d: d[2]["shape"][3], reverse=True)
    return by_register[0]