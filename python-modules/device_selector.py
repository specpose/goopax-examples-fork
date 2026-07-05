def retrieve_shape(memoryview_descriptor):
    #descriptor = (memoryview_descriptor[0][0],memoryview_descriptor[0][1][0][0],memoryview_descriptor[0][1][0][1][1],memoryview_descriptor[0][1][0][1][2])
    descriptor = (memoryview_descriptor[1][0],memoryview_descriptor[1][1][0][0],memoryview_descriptor[1][1][0][1][1],memoryview_descriptor[1][1][0][1][2])
    #descriptor = (memoryview_descriptor[1][0],memoryview_descriptor[1][1][1][0],memoryview_descriptor[1][1][1][1][1],memoryview_descriptor[1][1][1][1][2])
    return descriptor