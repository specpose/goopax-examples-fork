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
#include <algorithm>
#include <array>
#include <string>
#include <iostream>
template<typename T>
PyObject* convert_descriptor_item(T& arr) {
    auto pos = std::find(std::begin(arr),std::end(arr),0);
    std::cout << "pos: " << std::distance(std::begin(arr), pos) << std::endl;
    PyObject* result = PyTuple_New(std::distance(std::begin(arr), pos));
    for (std::size_t i = 0; i < std::distance(std::begin(arr), pos); ++i) {
        PyObject* element = PyLong_FromLong(arr[i]);
        PyTuple_SetItem(result, i, element);
    }
    return result;
}
static PyObject* memoryview_descriptor(PyObject* self, std::string format, std::array<Py_ssize_t, PyBUF_MAX_NDIM>& shape, std::array<Py_ssize_t, PyBUF_MAX_NDIM>& strides){
    PyObject* result = PyTuple_New(3);
    printf("%x: MemoryView Descriptor created\n",result);
    PyObject* f = PyUnicode_FromString(format.c_str());
    PyTuple_SetItem(result, 0, f);
    PyTuple_SetItem(result, 1, convert_descriptor_item(shape));
    PyTuple_SetItem(result, 2, convert_descriptor_item(strides));
    return result;
}

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
    for (size_t i = 0; i < PyBUF_MAX_NDIM; ++i) {
        buffy->shape[i] = 0;
        buffy->strides[i] = 0;
    }
    strcpy(buffy->format, "x");
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

namespace std
{
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& s, const std::array<T, N>& a)
{
    s << "(";
    for (size_t k = 0; k < N; ++k)
    {
        if (k != 0)
        {
            s << ", ";
        }
        s << a[k];
    }
    s << ")";
    return s;
}
}
#include <goopax_extra/types.hpp>
#include <map>
using namespace goopax;
static constexpr const char* backend_names[] = { "CPU", "OpenCL", "CUDA", "Metal", "Vulkan" };
const std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> python_types() {
    std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> known_types{};
    known_types[static_cast<type_enum_t>(10 * num_type_modes)].push_back(std::pair(std::string("d"),8));
    known_types[static_cast<type_enum_t>(3 * num_type_modes)].push_back(std::pair(std::string("h"),2));
    return known_types;
};
const std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> register_types(goopax_device& d) {
    auto type_to_python = python_types();
    std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> registered_type{};
    for (unsigned int t = 0; t < std::size(type_enum_raw_names); ++t)
    {
        const auto enumerated_type = static_cast<type_enum_t>(t * num_type_modes);
        if (d.support_type(enumerated_type))
        {
            if (type_to_python.find(enumerated_type) != end(type_to_python))
                registered_type[enumerated_type] = type_to_python[enumerated_type];
        }
    }
    return registered_type;
};
bool test_dev(unsigned int env) {
    goopax_device d = goopax::default_device(static_cast<goopax::envmode>(env));
    if (d.valid())
        return true;
    unsigned int log_env = env >> int(std::log2(env));
    return false;
}
std::vector<goopax_device> test_env(unsigned int env) {
    std::vector<goopax_device> dvec{};
    if (test_dev(env))
        dvec = goopax::devices(static_cast<goopax::envmode>(env));
    return dvec;
}
static Py_ssize_t queue_variables = 3;
static Py_ssize_t user_variables = 1;
std::array<Py_ssize_t, PyBUF_MAX_NDIM> check_shape(goopax_device& d, const Py_ssize_t len) {
    auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    std::fill(std::begin(shape), std::end(shape), 0);

    if (d.max_registers() == 0) {
        shape[3] = 1;
    } else {
        if (len % (d.max_registers() - queue_variables - user_variables) != 0) {
            shape[3] = 1;
        } else {
            shape[3] = d.max_registers() - queue_variables - user_variables;
        }
    }

    if (d.default_global_size_min() % d.default_local_size() != 0 || len / d.default_global_size_min() < 1) {
        shape[0] = 1;
    } else {
        shape[0] = d.default_global_size_min() / d.default_local_size();
    }

    if (len / d.default_local_size() < 1) {
        shape[1] = 1;
    } else {
        shape[1] = d.default_local_size();
    }

    if (len / shape[0] / shape[1] / shape[3] < 1) {
        shape[2] = 1;
    } else {
        shape[2] = len / (shape[0] * shape[1] * shape[3]);
    }

    return shape;
};
std::array<Py_ssize_t, PyBUF_MAX_NDIM> make_strides(const std::array<Py_ssize_t, PyBUF_MAX_NDIM>& shape, const std::string format, const Py_ssize_t itemsize) {
    auto strides = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    std::copy(begin(shape), end(shape), begin(strides));
    Py_ssize_t previous_stridesize = itemsize;
    std::for_each(std::rbegin(strides), std::rend(strides), [&previous_stridesize](auto& current){
        if (current != 0) {
            auto shape = current;
            current = previous_stridesize;
            previous_stridesize = shape * previous_stridesize;
        }
    });
    return strides;
};
static PyObject* gpu_reshaper_test_memoryview(PyObject* self, PyObject* args){
    PyObject* result = NULL;
    PyObject* ndarray = NULL;
    PyArg_ParseTuple(args, "O", &ndarray);
    printf("%x: Retrieving ndarray from tuple\n",ndarray);
    Py_buffer view;
    if (PyObject_GetBuffer(ndarray,&view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)==0){
        goopax::envmode env = static_cast<goopax::envmode>(1 << 4);
        std::vector<goopax_device> dvec = test_env(env);
        if (!dvec.empty()){
            printf("Got view format: %s, and len: %d\n", view.format, view.len);
            const auto shape = (std::array<Py_ssize_t, PyBUF_MAX_NDIM>*)view.shape;
            std::cout << *shape << std::endl;
            const auto strides = (std::array<Py_ssize_t, PyBUF_MAX_NDIM>*)view.strides;
            std::cout << *strides << std::endl;
            auto registered_type = register_types(dvec[0]);
            bool valid_format = false;
            for (const auto& ts : registered_type){
                for (const auto& p : ts.second)
                    if (p.first==view.format){
                        valid_format = true;
                        break;
                    }
                if (valid_format)
                    break;
            }
            auto gpu_shape = check_shape(dvec[0], view.len/2);
                std::cout << gpu_shape << std::endl;
            auto gpu_strides = make_strides(gpu_shape,std::string("h"),2);
                std::cout << gpu_strides << std::endl;
            if (!(valid_format && *shape==gpu_shape && *strides==gpu_strides))
                result = memoryview_descriptor(self, std::string(view.format),gpu_shape,gpu_strides);
            PyBuffer_Release(&view);
        }
    } else {
        abort();
    }
    return result;
}

static PyMethodDef gpu_reshaper_methods[] = {
    {"process_memoryview", gpu_reshaper_process_memoryview, METH_VARARGS, ""},
    {"test_memoryview", gpu_reshaper_test_memoryview, METH_VARARGS, ""},
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
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys,"path");
    PyObject* append = PyObject_GetAttrString(path,"append");
    PyObject* append_args = PyTuple_New(1);
    PyObject* directory = PyUnicode_FromString("../pyaudio-test");
    PyTuple_SetItem(append_args, 0, directory);
    PyObject_CallObject(append,append_args);

    PyObject* numpy_test = PyImport_ImportModule("numpy_test");
    PyObject* import_data = PyObject_GetAttrString(numpy_test,"import_data");
    PyObject* import_data_args = PyTuple_New(1);
    PyObject* filename = PyUnicode_FromString("../output.wav");
    PyTuple_SetItem(import_data_args, 0, filename);
    PyObject* numpy_array = PyObject_CallObject(import_data,import_data_args);
    printf("%x: Got a numpy_array.\n",numpy_array);


    PyObject* gpu_reshaper = PyImport_ImportModule("gpu_reshaper");
    PyObject* test = PyObject_GetAttrString(gpu_reshaper,"test_memoryview");
    PyObject* test_args = PyTuple_New(1);
    PyTuple_SetItem(test_args, 0, numpy_array);
    PyObject* test_object = PyObject_CallObject(test,test_args);

    //PyTypeObject *MyType = (PyTypeObject*)PyType_FromSpec(&MyObject_Spec);
    //PyObject* myobj = PyObject_CallObject((PyObject*)MyType,NULL);
    //PyObject_Del((void*)myobj);
    PyGILState_Release(gstate);
#if DEBUG
    if (Py_FinalizeEx() < 0)
        abort();
#else
    Py_Finalize();
#endif
    return 0;
}
#endif

#if DEBUG
#undef Py_DECREF
#define Py_DECREF Py_DECREF_bak;
#endif