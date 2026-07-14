#include <Python.h>
#include <stdio.h>

typedef short T;

#include <functional>
#include <goopax_extra/types.hpp>
using namespace goopax;
static const Py_ssize_t queue_variables = 3;
static const Py_ssize_t user_variables = 1;
class silence : public std::function<void(resource<T>&,gpu_uint)> {
    public:
        static const Py_ssize_t queue_variables = 3;
        static const Py_ssize_t user_variables = 1;
        silence(goopax_device& d) : std::function<void(resource<T>&,gpu_uint)>{}, _shape3(d.max_registers() - queue_variables - user_variables) {}
    void operator()(resource<T>& buffy, gpu_uint shape2){
        // Does not work for GOOPAX_ENV=1
        gpu_for(0+global_id()*_shape3, buffy.size(), global_size()*_shape3, [&buffy,this](gpu_uint k){
            for(unsigned int l = 0; l < _shape3; ++l) {
                buffy[k+l] -= 257; // T
            }
        });
        //gpu_for_group(0, num_groups(), [&buffy,shape2,this](gpu_uint i){
        //    gpu_for_local(0, local_size(), [&buffy,shape2,this](gpu_uint j){
        //        gpu_for(0, shape2, [&buffy,shape2,this](gpu_uint k){
        //            for (unsigned int l = 0; l < _shape3; ++l) {
        //                buffy[global_id()*shape2*_shape3+k*_shape3+l] -= 257; // T
        //            }
        //        });
        //    });
        //});
    };
    private:
    unsigned int _shape3;
};

/*struct silence {
    silence(goopax_device& d) : _shape3(d.max_registers() - queue_variables - user_variables) {}
    unsigned int _shape3;
    static const unsigned int queue_variables = 3;
    static const unsigned int user_variables = 1;
    std::function<void(resource<T>&,gpu_uint)> afunc = [&](resource<T>& buffy, gpu_uint shape2) {
    };
};*/

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

std::array<Py_ssize_t, PyBUF_MAX_NDIM> safe_copy(Py_ssize_t* first, Py_ssize_t len) {
    std::array<Py_ssize_t, PyBUF_MAX_NDIM> copy{};
    std::fill(std::begin(copy), std::end(copy), 0);
    for (Py_ssize_t i = 0; i < len; ++i)
        copy[i] = *(first + i);
    return copy;
};

#include <algorithm>
#include <array>
#include <string>
#include <iostream>
template<typename T>
PyObject* convert_descriptor_item(T& arr) {
    auto pos = std::find(std::begin(arr),std::end(arr),0);
    PyObject* result = PyTuple_New(std::distance(std::begin(arr), pos));
    for (std::size_t i = 0; i < std::distance(std::begin(arr), pos); ++i) {
        PyObject* element = PyLong_FromLong(arr[i]);
        PyTuple_SetItem(result, i, element);
    }
    return result;
}
static PyObject* memoryview_descriptor(std::string format, std::array<Py_ssize_t, PyBUF_MAX_NDIM>& shape, std::size_t alignment){
    PyObject* result = PyTuple_New(3);
    PyObject* f = PyUnicode_FromString(format.c_str());
    PyTuple_SetItem(result, 0, f);
    PyTuple_SetItem(result, 1, convert_descriptor_item(shape));
    PyTuple_SetItem(result, 2, PyLong_FromLong(alignment));
    return result;
}

#include <map>
#include "gpu_reshaper.hpp"
PyObject* host_query(std::function<std::array<Py_ssize_t, PyBUF_MAX_NDIM>(goopax::goopax_device&, const Py_ssize_t)> Check_shape, PyObject* empty, PyObject* args) {
    PyObject* result = PyList_New(0);
    const char* format = NULL;
    Py_ssize_t format_length = 0;
    Py_ssize_t length = 0;
    if (PyArg_ParseTuple(args, "s#n", &format, &format_length, &length)) {
        printf("Got format: %s, and len: %d\n", format, length);
        for (unsigned int log_env = 0; log_env < 10; ++log_env) {
            goopax::envmode env = static_cast<goopax::envmode>(1 << log_env);
            std::vector<goopax_device> dvec = test_env(env);
            if (!dvec.empty()){
                PyObject* Env = PyTuple_New(2);
                PyTuple_SetItem(Env, 0, PyLong_FromLong(log_env));
                PyObject* Devices = PyList_New(0);
                for (size_t device_i = 0; device_i < dvec.size(); ++device_i) {
                    //if (!(*shape==gpu_shape && *strides==gpu_strides))
                    auto registered_type = register_types(dvec[device_i]);
                    for (const auto& dev_ts : registered_type){
                    for (const auto& t : dev_ts.second){
                    if (t.first == format && itemsize(t.first) == itemsize(format)) {
                        PyObject* Device = PyTuple_New(2);
                        const std::array< uint8_t, 16 > device_id = dvec[device_i].uuid();
                        PyTuple_SetItem(Device, 0, PyBytes_FromStringAndSize(reinterpret_cast<const char*>(&device_id), std::size(device_id)));
                        auto gpu_shape = Check_shape(dvec[device_i], length / itemsize(format));
                        //std::cout << gpu_shape << std::endl;
                        auto gpu_strides = make_strides(gpu_shape, itemsize(format));
                        //std::cout << gpu_strides << std::endl;
                        PyObject* descr = memoryview_descriptor(std::string(format),gpu_shape, dvec[device_i].min_host_ptr_alignment());
                        PyTuple_SetItem(Device, 1, descr);
                        PyList_Append(Devices, Device);
                    }
                    }
                    }
                    }
                    PyTuple_SetItem(Env, 1, Devices);
                    PyList_Append(result, Env);
                }
            }
    } else {
        abort();
    }
    return result;
}
static PyObject* SimdFunction_host_query(PyObject* empty, PyObject* args){
    return host_query(SimdCheck_shape, empty, args);
}

/*  Type    */
template<typename FunctionClass> struct SimdFunction{
    PyObject_VAR_HEAD
    goopax_device* device;
    FunctionClass* func;
};
void SimdFunction_dealloc(PyObject* self) {
    printf("Dealloc\n");
    Py_TYPE(self)->tp_free(self);
};
template<typename FunctionClass> PyObject* SimdFunction_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds){
    printf("New\n");
    SimdFunction<FunctionClass>* obj = (SimdFunction<FunctionClass>*)subtype->tp_alloc(subtype, 0);
    if (obj != NULL) {
        obj->device = (goopax_device*)malloc(sizeof(goopax_device));
        //obj->func = (FunctionClass*)malloc(sizeof(FunctionClass));
    }
    return (PyObject*)obj;
};
template<typename FunctionClass> void SimdFunction_free(void *self) {
    printf("Free\n");
    SimdFunction<FunctionClass>* obj = (SimdFunction<FunctionClass>*)self;
    free((void*)obj->device);
    free((void*)obj->func);
};
template<typename FunctionClass> int SimdFunction_init(PyObject* self, PyObject* args, PyObject* kwds) {
    printf("Init\n");
    SimdFunction<FunctionClass>* _self = (SimdFunction<FunctionClass>*) self;
    int log_env = 0;
    const char* gpu_uuid = NULL;
    Py_ssize_t gpu_uuid_length = 0;
    if (PyArg_ParseTuple(args, "iy#", &log_env, &gpu_uuid, &gpu_uuid_length)) {
        const Py_ssize_t d_uuid_length = std::tuple_size<std::array< uint8_t, 16 >>{};
        if (gpu_uuid_length!=d_uuid_length)
            abort();

        envmode env = static_cast<envmode>(1 << log_env);
        std::vector<goopax_device> dvec = devices(static_cast<goopax::envmode>(env));
        for (size_t device_i = 0; device_i < dvec.size(); ++device_i) {
            const std::array< uint8_t, d_uuid_length >& d_uuid= dvec[device_i].uuid();
            if (memcmp(gpu_uuid, d_uuid.data(), d_uuid_length) == 0) {
                *(_self->device) = dvec[device_i];
                _self->func = new FunctionClass(dvec[device_i]);
                printf("Init Success\n");
                return 0;
            }
        }
    }
    _self->device = nullptr;
    _self->func = nullptr;
    return -1;
};
#include <goopax_extra/output.hpp>
template<typename FunctionClass> PyObject* SimdFunction_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    printf("Call\n");
    SimdFunction<FunctionClass>* _self = (SimdFunction<FunctionClass>*) self;
    PyObject* mvobject = NULL;

    if (PyArg_ParseTuple(args, "O", &mvobject)) {
        if (PyMemoryView_Check(mvobject)!=1)
            abort();
        //Py_buffer* view = PyMemoryView_GET_BUFFER(mvobject);
        //PyObject* obj = PyMemoryView_GET_BASE(mvobject);
        Py_buffer* view = new Py_buffer;
        if (PyObject_GetBuffer(mvobject, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE)==0){
        printf("Processing memoryview with: view.len %d, view.itemsize %d.\n", view->len, view->itemsize);
        auto shape = safe_copy(view->shape, view->ndim);
        auto strides = safe_copy(view->strides, view->ndim);
        unsigned int shape0 = shape[0];
        unsigned int shape1 = shape[1];
        unsigned int shape2 = shape[2];
        unsigned int shape3 = shape[3];
        std::size_t device_alignment = _self->device->min_host_ptr_alignment();
        printf("Device min host ptr alignment: %d\n", device_alignment);
        std::size_t padded = (view->len + device_alignment - 1) & ~(device_alignment - 1);
        std::cout << "padded " << padded << " len " << view->len << " shape0 " << shape0 << " shape1 " << shape1 << " shape2 " << shape2 << " shape3 " << shape3 << " ptr " << view->buf << std::endl;
        buffer<T> buffy(*(_self->device), view->len/view->itemsize, (T*)view->buf, BUFFER_READ_WRITE);
        kernel<void(buffer<T>&,unsigned int)> process(*(_self->device), *(_self->func), shape1, shape0*shape1);
        //kernel<void(buffer<T>&,unsigned int)> process(*(_self->device), _self->func->afunc, shape1, shape0*shape1);
        process(buffy,shape2);
        _self->device->wait_all();
        //for(std::size_t i=0; i<shape[0]*shape[1]*shape[2]*shape[3]; ++i)
        //    printf("%d,",((T*)view->buf)[i]);
        //std::cout<< buffy << std::endl;
        PyBuffer_Release(view);
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
};

PyTypeObject SimdFunction_Type(
    const char* SimdFunction_name,
    const char* SimdFunction_doc_string,
    Py_ssize_t basic_size,
    PyMethodDef* methods,
    PyObject* (*python_new_function)(PyTypeObject*, PyObject*, PyObject*),
    int (*python_init_function)(PyObject*, PyObject*, PyObject*),
    PyObject* (*python_call_function)(PyObject*, PyObject*, PyObject*),
    void (*python_free_function)(void*)
) {
    PyTypeObject type = {
    PyVarObject_HEAD_INIT(NULL, 0) // typing.Callable
    SimdFunction_name,
    basic_size,
    0,
    (destructor)SimdFunction_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    (ternaryfunc)python_call_function, // SimdFunction_call
    0, 0, 0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    PyDoc_STR(SimdFunction_doc_string),
    0, 0, 0, 0, 0, 0,
    methods,
    0, 0, 0, 0, 0, 0, 0,
    (initproc)python_init_function,
    PyType_GenericAlloc,
    (newfunc)python_new_function,
    (freefunc)python_free_function,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    return type;
};


/*  Module  */
static struct PyModuleDef gpu_reshaper_module = {
    PyModuleDef_HEAD_INIT,
    "gpu_reshaper",
    "",
    0,
    0, // methods
    NULL,
    NULL,
    NULL,
    NULL
};
static PyMethodDef silence_py_methods[] = {
    {"host_query", SimdFunction_host_query, METH_VARARGS | METH_STATIC, ""},
    {NULL, NULL, 0, NULL}
};
static PyTypeObject silence_type_descr = SimdFunction_Type("silence","Sets all the samples to zero",sizeof(SimdFunction<silence>),silence_py_methods,SimdFunction_new<silence>,SimdFunction_init<silence>,SimdFunction_call<silence>,SimdFunction_free<silence>);
PyMODINIT_FUNC PyInit_gpu_reshaper(void){
    PyObject* mod = PyModule_Create(&gpu_reshaper_module);
    if (PyModule_AddType(mod, &silence_type_descr) < 0)
        abort();
    return mod;
}

#if !MODULE_LIBRARY
int main(){
    const char* arg1 = "../output.wav";

    PyImport_AppendInittab("gpu_reshaper", PyInit_gpu_reshaper);
    Py_InitializeEx(0);
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* sys_m = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys_m,"path");
    PyObject* append = PyObject_GetAttrString(path,"append");
    PyObject* append_args = PyTuple_New(1);
    PyObject* directory = PyUnicode_FromString("../python-modules");
    PyTuple_SetItem(append_args, 0, directory);
    PyObject_CallObject(append,append_args);

    PyObject* numpy_test_m = PyImport_ImportModule("numpy_test");

    PyObject* get_format = PyObject_GetAttrString(numpy_test_m,"wav_format");
    PyObject* get_format_args = PyTuple_New(1);
    PyObject* filename = PyUnicode_FromString(arg1);
    PyTuple_SetItem(get_format_args, 0, filename);
    PyObject* wav_format = PyObject_CallObject(get_format,get_format_args);
    PyObject_Print(wav_format,stdout,Py_PRINT_RAW);

    PyObject* gpu_reshaper_m = PyImport_ImportModule("gpu_reshaper");
    PyObject* silence = PyObject_GetAttrString(gpu_reshaper_m,"silence");
    PyObject* descriptor = PyObject_CallMethod(silence,"host_query","O",wav_format);
    printf("Dumping memoryview descriptor:\n");
    PyObject_Print(descriptor,stdout,Py_PRINT_RAW);
    printf("\n");

    if (PyList_Size(descriptor)>0){
        PyObject* device_selector_m = PyImport_ImportModule("device_selector");
        PyObject* select = PyObject_GetAttrString(device_selector_m,"retrieve_shape");
        PyObject* select_args = PyTuple_New(1);
        PyTuple_SetItem(select_args, 0, descriptor);
        PyObject* selection = PyObject_CallObject(select,select_args);
        PyObject_Print(selection,stdout,Py_PRINT_RAW);

        PyObject* import_data = PyObject_GetAttrString(numpy_test_m,"import_wav_aligned");
        PyObject* import_data_args = PyTuple_New(3);
        PyTuple_SetItem(import_data_args, 0, filename);
        PyTuple_SetItem(import_data_args, 1, wav_format);
        PyTuple_SetItem(import_data_args, 2, selection);
        PyObject* numpy_array = PyObject_CallObject(import_data,import_data_args);
        printf("Imported data from file \"%s\".\n", arg1);

        //PyObject* shape = PyTuple_GetItem(selection, 2);
        //PyObject* reshape = PyObject_GetAttrString(numpy_array,"reshape");
        //printf("Python ndarray reshaping to:\n");
        //PyObject_Print(shape,stdout,Py_PRINT_RAW);
        //printf("\n");
        //PyObject* shape_args = PyTuple_New(1);
        //PyTuple_SetItem(shape_args, 0, shape);
        //numpy_array = PyObject_CallObject(reshape, shape_args);

        PyObject* Initialisation = PyTuple_New(2);
        PyTuple_SetItem(Initialisation, 0, PyTuple_GetItem(selection, 0));
        PyTuple_SetItem(Initialisation, 1, PyTuple_GetItem(selection, 1));
        PyObject* Custom_Callable = PyObject_CallObject(silence, Initialisation);

        //Py_buffer* view;
        //if (PyObject_GetBuffer(numpy_array, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)==0){
        PyObject* process_args = PyTuple_New(1);
        PyObject* mv = PyMemoryView_GetContiguous(numpy_array, PyBUF_WRITE, 'C');
        //PyObject* mv = PyMemoryView_FromObject(numpy_array);
        //PyObject* mv = PyMemoryView_FromBuffer(view);
        PyTuple_SetItem(process_args, 0, mv);
        PyObject* result = PyObject_CallObject(Custom_Callable, process_args);
        PyObject_Del((void*)Custom_Callable);
        //PyBuffer_Release(view);
        //}

        printf("Exporting data to file \"%s\".\n", arg1);
        PyObject* export_data = PyObject_GetAttrString(numpy_test_m,"export_data");
        PyObject* export_data_args = PyTuple_New(2);
        PyTuple_SetItem(export_data_args, 0, filename);
        Py_INCREF(numpy_array);
        PyTuple_SetItem(export_data_args, 1, numpy_array);
        PyObject_CallObject(export_data,export_data_args);
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