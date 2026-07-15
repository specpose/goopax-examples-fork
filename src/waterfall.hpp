#include <functional>
using namespace goopax;

/*  GPU  */
std::vector<goopax::goopax_device> test_env(unsigned int env) {
    std::vector<goopax::goopax_device> dvec{};
    if (test_dev(env))
        dvec = goopax::devices(static_cast<goopax::envmode>(env));
    return dvec;
}
static constexpr const char* backend_names[] = { "CPU", "OpenCL", "CUDA", "Metal", "Vulkan" };
std::string type_to_python_name(goopax::type_enum_t type) { return goopax::type_enum_raw_names[type / goopax::num_type_modes]; };
/*  https://docs.python.org/3/library/struct.html#format-characters */
/*  https://numpy.org/doc/stable/reference/arrays.interface.html#arrays-interface   */
static std::map<goopax::type_enum_t, std::map<std::string, std::string>> known_types {
    {goopax::type_enum<double>::value, { {std::string("d"),std::string("f")} }},
    {goopax::type_enum<int16_t>::value, { {std::string("h"),std::string("i")} }}
};
const std::map<goopax::type_enum_t, std::map<std::string, std::string>> register_types(goopax::goopax_device& d) {
    std::map<goopax::type_enum_t, std::map<std::string, std::string>> registered_type{};
    for (unsigned int t = 0; t < std::size(goopax::type_enum_raw_names); ++t)
    {
        const auto enumerated_type = static_cast<goopax::type_enum_t>(t * goopax::num_type_modes);
        if (d.support_type(enumerated_type))
        {
            if (known_types.find(enumerated_type) != end(known_types))
                registered_type[enumerated_type] = known_types[enumerated_type];
        }
    }
    return registered_type;
};

/* Util */
std::array<Py_ssize_t, PyBUF_MAX_NDIM> safe_copy(Py_ssize_t* first, Py_ssize_t len) {
    std::array<Py_ssize_t, PyBUF_MAX_NDIM> copy{};
    std::fill(std::begin(copy), std::end(copy), 0);
    for (Py_ssize_t i = 0; i < len; ++i)
        copy[i] = *(first + i);
    return copy;
};
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

/* Interface */
PyObject* host_query(std::function<std::array<Py_ssize_t, PyBUF_MAX_NDIM>(goopax_device&, const Py_ssize_t)> Check_shape, PyObject* empty, PyObject* args) {
    PyObject* result = PyList_New(0);
    const char* format = NULL;
    Py_ssize_t format_length = 0;
    Py_ssize_t length = 0;
    if (PyArg_ParseTuple(args, "s#n", &format, &format_length, &length)) {
        printf("Got format: %s, and len: %d\n", format, length);
        for (unsigned int log_env = 0; log_env < 10; ++log_env) {
            envmode env = static_cast<envmode>(1 << log_env);
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

/*  Interface: Types    */
template<typename FunctionClass> std::array<Py_ssize_t, PyBUF_MAX_NDIM> SimdCheck_shape(goopax::goopax_device& d, const Py_ssize_t len) {
    auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    std::fill(std::begin(shape), std::end(shape), 0);

    /*  Explicit Unrolling  */
    if (d.max_registers() == 0) {
        shape[3] = FunctionClass::queue_variables + FunctionClass::user_variables;
    } else {
        if (len % (d.max_registers() - FunctionClass::queue_variables - FunctionClass::user_variables) != 0) {
            shape[3] = 1;
        } else {
            shape[3] = d.max_registers() - FunctionClass::queue_variables - FunctionClass::user_variables;
        }
    }

    /*  Workgroups  */
    if (len / d.default_num_groups_min() < 1) {
        shape[0] = 1;
    } else {
        shape[0] = d.default_num_groups_min();
    }

    /*  ALUs  */
    if (len / d.default_local_size() < 1) {
        shape[1] = 1;
    } else {
        shape[1] = d.default_local_size();
    }

    /*  Queue  */
    if (len / shape[0] / shape[1] / shape[3] < 1) {
        shape[2] = 1;
    } else {
        shape[2] = len / (shape[0] * shape[1] * shape[3]);
    }

    return shape;
};
template<typename FunctionClass> struct SimdFunction{
    PyObject_VAR_HEAD
    goopax_device device;
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
        obj->func = (FunctionClass*)malloc(sizeof(FunctionClass));
    }
    return (PyObject*)obj;
};
template<typename FunctionClass> void SimdFunction_free(void *self) {
    printf("Free\n");
    SimdFunction<FunctionClass>* obj = (SimdFunction<FunctionClass>*)self;
    free((void*)obj->func);
};
template<typename FunctionClass> int SimdFunction_init(PyObject* self, PyObject* args, PyObject* kwds) {
    printf("Init\n");
    SimdFunction<FunctionClass>* _self = (SimdFunction<FunctionClass>*) self;
    int log_env = 0;
    const unsigned char* gpu_uuid = NULL;
    Py_ssize_t gpu_uuid_length = 0;
    if (PyArg_ParseTuple(args, "iy#", &log_env, &gpu_uuid, &gpu_uuid_length)) {
        const Py_ssize_t d_uuid_length = std::tuple_size<std::array< uint8_t, 16 >>{};
        printf("log_env is %d.\n",log_env);
        printf("uuid_length is %d.\n",gpu_uuid_length);
        if (gpu_uuid_length!=d_uuid_length)
            abort();

        envmode env = static_cast<envmode>(1 << log_env);
        std::vector<goopax_device> dvec = devices(static_cast<envmode>(env));
        for (size_t device_i = 0; device_i < dvec.size(); ++device_i) {
            const std::array< uint8_t, d_uuid_length >& d_uuid= dvec[device_i].uuid();
            for(std::size_t i = 0; i < gpu_uuid_length; ++i)
                printf("%x,",gpu_uuid[i]);
            printf("\n");
            for(std::size_t i = 0; i < d_uuid_length; ++i)
                printf("%x,",gpu_uuid[i]);
            printf("\n");
            if (memcmp(gpu_uuid, d_uuid.data(), d_uuid_length) == 0) {
                _self->device = dvec[device_i];
                dvec[device_i];
                _self->func = new FunctionClass(dvec[device_i]);
                printf("Init Success\n");
                return 0;
            }
        }
    }
    _self->func = nullptr;
    return -1;
};
//#include <goopax_extra/output.hpp>
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
            std::size_t device_alignment = _self->device.min_host_ptr_alignment();
            printf("Device min host ptr alignment: %d\n", device_alignment);
            std::size_t padded = (view->len + device_alignment - 1) & ~(device_alignment - 1);
            std::cout << "padded " << padded << " len " << view->len << " shape0 " << shape0 << " shape1 " << shape1 << " shape2 " << shape2 << " shape3 " << shape3 << " ptr " << view->buf << std::endl;
            buffer<T> buffy(_self->device, view->len/view->itemsize, (T*)view->buf, BUFFER_READ_WRITE);
            kernel<void(buffer<T>&,unsigned int)> process(_self->device, *(_self->func), shape1, shape0*shape1);
            //kernel<void(buffer<T>&,unsigned int)> process(*(_self->device), _self->func->afunc, shape1, shape0*shape1);
            process(buffy,shape2);
            _self->device.wait_all();
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