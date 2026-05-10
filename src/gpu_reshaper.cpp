#include <Python.h>
#include <stdio.h>

typedef short T;

static void process(void* buffer, Py_ssize_t* strides, Py_ssize_t* shape){
    T* _buffer = (T*)buffer;
    for (Py_ssize_t i=0; i<shape[0]; ++i)
        for (Py_ssize_t j=0; j<shape[1]; ++j)
            for (Py_ssize_t k=0; k<shape[2]; ++k)
                for (Py_ssize_t l=0; l<shape[3]; ++l)
                    printf("%d,",_buffer[(i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3])/sizeof(T)]);
    printf("\n");
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
    if (PyArg_ParseTuple(args, "O", &mvobject)) {
    printf("%x: Retrieving memoryview from tuple\n",mvobject);
    if (PyMemoryView_Check(mvobject)!=1)
        abort();
    Py_buffer* view = PyMemoryView_GET_BUFFER(mvobject);
    printf("view.len: %d, view.itemsize: %d\n",view->len,view->itemsize);
    //process(view->buf,view->strides,view->shape);
    }
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
std::string type_to_python_name(type_enum_t type) { return type_enum_raw_names[type / num_type_modes]; };
static std::map<type_enum_t, std::map<std::string, Py_ssize_t>> known_types {
    {type_enum<double>::value, { {std::string("d"),8} }},
    {type_enum<int16_t>::value, { {std::string("h"),2} }}
};
const std::map<type_enum_t, std::map<std::string, Py_ssize_t>> register_types(goopax_device& d) {
    std::map<type_enum_t, std::map<std::string, Py_ssize_t>> registered_type{};
    for (unsigned int t = 0; t < std::size(type_enum_raw_names); ++t)
    {
        const auto enumerated_type = static_cast<type_enum_t>(t * num_type_modes);
        if (d.support_type(enumerated_type))
        {
            if (known_types.find(enumerated_type) != end(known_types))
                registered_type[enumerated_type] = known_types[enumerated_type];
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
#include <iomanip>
static PyObject* gpu_reshaper_test_memoryview(PyObject* self, PyObject* args){
    PyObject* result = PyList_New(0);
    PyObject* ndarray = NULL;
    if (PyArg_ParseTuple(args, "O", &ndarray)) {
    printf("%x: Retrieving ndarray from tuple\n",ndarray);
    Py_buffer view;
    if (PyObject_GetBuffer(ndarray,&view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)==0){
        goopax::envmode env = static_cast<goopax::envmode>(1 << 4);
        std::vector<goopax_device> dvec = test_env(env);
        PyObject* Env = PyTuple_New(2);
        PyTuple_SetItem(Env, 0, PyLong_FromLong(4));
        if (!dvec.empty()){
            PyObject* Device = PyTuple_New(2);
            const std::array< uint8_t, 16 > device_id = dvec[0].uuid();
            PyTuple_SetItem(Device, 0, PyBytes_FromStringAndSize(reinterpret_cast<const char*>(&device_id), std::size(device_id)));

            /*printf("Got view format: %s, and len: %d\n", view.format, view.len);
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
            }*/
            auto gpu_shape = check_shape(dvec[0], view.len/2);
                std::cout << gpu_shape << std::endl;
            auto gpu_strides = make_strides(gpu_shape,std::string("h"),2);
                std::cout << gpu_strides << std::endl;
            //if (!(valid_format && *shape==gpu_shape && *strides==gpu_strides)){
                PyObject* descr = memoryview_descriptor(self, std::string(view.format),gpu_shape,gpu_strides);
                PyTuple_SetItem(Device, 1, descr);
            //}
            PyTuple_SetItem(Env, 1, Device);
        }
        //Py_INCREF(Env);
        PyList_Append(result, Env);
        PyBuffer_Release(&view);
    } else {
        abort();
    }
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
    PyObject* reshape = PyObject_GetAttrString(numpy_array,"reshape");

    PyObject* gpu_reshaper = PyImport_ImportModule("gpu_reshaper");
    PyObject* process_call = PyObject_GetAttrString(gpu_reshaper,"process_memoryview");
    PyObject* test = PyObject_GetAttrString(gpu_reshaper,"test_memoryview");
    PyObject* test_args = PyTuple_New(1);
    PyTuple_SetItem(test_args, 0, numpy_array);
    PyObject* descriptor = PyObject_CallObject(test,test_args);
    PyObject_Print(descriptor,stdout,Py_PRINT_RAW);
    if (PyList_Size(descriptor)>0){
        PyObject* env = PyList_GetItem(descriptor, 0);
        PyObject* device = PyTuple_GetItem(env, 1);
        PyObject* descr = PyTuple_GetItem(device,1);
        PyObject* shape = PyTuple_GetItem(descr,1);
        PyObject_Print(shape,stdout,Py_PRINT_RAW);
        PyObject* shape_args = PyTuple_New(1);
        PyTuple_SetItem(shape_args, 0, shape);

        numpy_array = PyObject_CallObject(reshape, shape_args);
        PyObject* process_args = PyTuple_New(1);
        PyTuple_SetItem(process_args, 0, PyMemoryView_FromObject(numpy_array));
        PyObject* result = PyObject_CallObject(process_call, process_args);
    }

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