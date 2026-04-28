#include <Python.h>
#include <stdio.h>

typedef double T;
static const T test_data[] = {0,0,1,0,1+sqrt(2)/2,sqrt(2)/2,1+sqrt(2)/2,1+sqrt(2)/2};

static void process(void* buffer, Py_ssize_t* strides, Py_ssize_t* shape){
    printf("shape=(%d,%d)\n",shape[0],shape[1]);
    T* _buffer = (T*)buffer;
    for (Py_ssize_t i=0; i<shape[0]*shape[1]; i+=2)
        printf("(%f,%f)\n",_buffer[i],_buffer[i+1]);
}

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API

#if DEBUG
#define Py_DECREF_bak Py_DECREF
void __delete(PyObject* X){
    //int* refcnt = (int*)(X);
    //*refcnt = 0;
    Py_DECREF_bak(X);
    if (Py_REFCNT(X) != 0)
        printf("%d\n",Py_REFCNT(X));
    if (Py_REFCNT(X) < 0)
        abort();
}
#undef Py_DECREF
#define Py_DECREF(X) __delete(X);
#endif

static PyObject* gpu_reshaper_process_memoryview(PyObject* self, PyObject* args){
    PyObject* mvobject = NULL;
    PyArg_ParseTuple(args, "O", &mvobject);
    printf("%x: Retrieving memoryview from tuple\n",mvobject);
    if (PyMemoryView_Check(mvobject)!=1)
        abort();
    Py_buffer* view = PyMemoryView_GET_BUFFER(mvobject);
    process(view->buf,view->strides,view->shape);
    Py_INCREF(Py_None);
    return Py_None;
}

#if !MODULE_LIBRARY
typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t* shape;
    Py_ssize_t* strides;
    char* format;
    T* ptr;
} MyObject;
static const char _format[] = "d"; // T
int get_buffer(PyObject *exporter, Py_buffer *view, int flags){
    printf("GetBuffer\n");
    if ((flags & (PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) == (PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) {
        view->ndim = 2;
        Py_ssize_t* shape = ((MyObject*)exporter)->shape;
        shape[0]=4;
        shape[1]=2;
        Py_ssize_t* strides = ((MyObject*)exporter)->strides;
        strides[0]=16;
        strides[1]=8;
        view->shape = shape;
        view->strides = strides;
        view->suboffsets = NULL;
        view->format = ((MyObject*)exporter)->format;
        strcpy(view->format, _format);
        view->itemsize = sizeof(T);
        view->len = view->shape[0]*view->shape[1]*view->itemsize;
        T* ptr = ((MyObject*)exporter)->ptr;
        view->readonly = 1;
        view->buf = (void*)ptr;
        view->obj = exporter;
    } else {
        abort();
    }
    return 0;
};
void release_buffer(PyObject *exporter, Py_buffer *view){
    printf("ReleaseBuffer\n");
};
PyObject *myobj_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds){
    printf("New\n");
    MyObject* buffy = (MyObject*)subtype->tp_alloc(subtype, 0);
    if (buffy != NULL) {
        buffy->shape = (Py_ssize_t*)malloc(PyBUF_MAX_NDIM*sizeof(Py_ssize_t));
        buffy->strides = (Py_ssize_t*)malloc(PyBUF_MAX_NDIM*sizeof(Py_ssize_t));
        buffy->format = (char*)malloc(sizeof(_format));
        buffy->ptr = (T*)malloc(sizeof(test_data)*sizeof(T));
    }
    return (PyObject*)buffy;
}
void myobj_free(void *self) {
    printf("Free\n");
    MyObject* buffy = (MyObject*)self;
    free((void*)buffy->shape);
    free((void*)buffy->strides);
    free((void*)buffy->format);
    free((void*)buffy->ptr);
}
int myobj_init(PyObject* self, PyObject* args, PyObject* kwds) {
    printf("Init\n");
    MyObject* buffy = (MyObject*)self;
    for (size_t i=0; i<sizeof(test_data)/sizeof(T); i++)
        buffy->ptr[i] = test_data[i];
    return 0;
}
static char _doc_string[] = "My objects";
static PyType_Slot MyObject_Slots[] = {
    {Py_tp_doc, _doc_string},
    {Py_tp_new, (void*)myobj_new},
    {Py_tp_init, (void*)myobj_init},
    {Py_tp_free, (void*)myobj_free},
    {Py_bf_getbuffer, (void*)get_buffer},
    {Py_bf_releasebuffer, (void*)release_buffer},
    {0, NULL}
};
static const char _name[] = "mymod.MyObject";
static PyType_Spec MyObject_Spec = {
    .name = _name,
    .basicsize = sizeof(MyObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = MyObject_Slots
};

static PyObject* gpu_reshaper_test_memoryview(PyObject* self, PyObject* what){
    PyTypeObject *MyType = (PyTypeObject*)PyType_FromSpec(&MyObject_Spec);
    PyObject* myobj = PyObject_CallObject((PyObject*)MyType,NULL);  // .tp_new
//    MyObject* myobj = PyObject_New(MyObject,MyType);
//    PyObject* myobj_ = PyObject_Init((PyObject*)myobj, MyType);
    Py_buffer view;
    if (PyObject_GetBuffer((PyObject*)myobj,&view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)==0){
        PyObject* mvobject = PyMemoryView_FromBuffer(&view);
        PyObject* printMemoryview = PyObject_GetAttrString(self,"process_memoryview");
        PyObject* args = PyTuple_New(1);
        printf("%x: New tuple with memoryview created\n",args);
        printf("%x: Passing memoryview to tuple\n",mvobject);
        PyTuple_SetItem(args, 0, mvobject);
        PyObject* result = PyObject_CallObject(printMemoryview, args);
        PyBuffer_Release(&view);
    }
    PyObject_Del((void*)myobj);
    return PyLong_FromLong(0);
}
#endif

static PyMethodDef gpu_reshaper_methods[] = {
    {"process_memoryview", gpu_reshaper_process_memoryview, METH_VARARGS, ""},
#if !MODULE_LIBRARY
    {"test_memoryview", gpu_reshaper_test_memoryview, METH_NOARGS, ""},
#endif
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gpu_reshaper_module = {
    PyModuleDef_HEAD_INIT,
    "gpu_reshaper",
    "",
    0,
    gpu_reshaper_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_gpu_reshaper(void){
    return PyModule_Create(&gpu_reshaper_module);
}

#if !MODULE_LIBRARY
int main(){
    PyImport_AppendInittab("gpu_reshaper", PyInit_gpu_reshaper);
    Py_InitializeEx(0);

    PyObject* unittest = PyImport_ImportModule("unittest");
    PyObject* tcase = PyObject_GetAttrString(unittest,"FunctionTestCase");
    PyObject* gpu_reshaper = PyImport_ImportModule("gpu_reshaper");
    PyObject* tm = PyObject_GetAttrString(gpu_reshaper,"test_memoryview");
    PyObject* tsuite = PyObject_GetAttrString(unittest,"TestSuite");
    PyObject* tresult = PyObject_GetAttrString(unittest,"TestResult");
    if (!tresult)
        printf("TestResult class was not found in unittest\n");
    PyObject* case2_args = PyTuple_New(1);
    PyTuple_SetItem(case2_args, 0, tm);
    PyObject* case2 = PyObject_CallObject(tcase,case2_args);
    PyObject* tsuite_args = PyTuple_New(1);
    PyObject* tests_tuple = PyTuple_New(1);
    PyTuple_SetItem(tests_tuple,0,case2);
    PyTuple_SetItem(tsuite_args,0,tests_tuple);
    PyObject* runner = PyObject_CallObject(tsuite, tsuite_args);
    PyObject* run_method = PyObject_GetAttrString(runner,"run");
    if (!run_method)
        printf("run_method not found\n");
    PyObject* result = PyObject_CallObject(tresult,NULL);
    if (!result)
        printf("Result was not instantiated\n");
    PyObject* run_args = PyTuple_New(1);
    PyTuple_SetItem(run_args,0,result);
    PyObject* run = PyObject_CallObject(run_method,run_args);
    if (!run)
        abort();
    PyObject* times = PyObject_GetAttrString(result,"collectedDurations");
    for (int i=0;i<PyList_Size(times);i++) {
        PyObject* tindex = PyList_GetItem(times,i);
        printf("%s took %f.\n",PyTuple_GetItem(tindex,0),PyFloat_AsDouble(PyTuple_GetItem(tindex,1)));
    }
    int uS = PyList_Size(PyObject_GetAttrString(result,"unexpectedSuccesses"));
    int f = PyList_Size(PyObject_GetAttrString(result,"failures"));
    int e = PyList_Size(PyObject_GetAttrString(result,"errors"));
    long tR = PyLong_AsLong(PyObject_GetAttrString(result,"testsRun"));
#if DEBUG
    if (Py_FinalizeEx() < 0)
        abort();
#else
    Py_Finalize();
#endif
    return tR>0&&e==0&&f==0&&uS==0 ? 0 : 1;
}
#endif

#if DEBUG
#undef Py_DECREF
#define Py_DECREF Py_DECREF_bak;
#endif