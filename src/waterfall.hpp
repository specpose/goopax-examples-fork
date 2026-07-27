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
unsigned long long lcm(unsigned long a, unsigned long b) {
    unsigned long& larger = a;
    unsigned long& smaller = b;
    if (a<b){
        larger = b;
        smaller = a;
    }
    unsigned int n = 1;
    while (n*smaller%larger!=0) {
        ++n;
    }
    return n*smaller;
}
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
static PyObject* memoryview_descriptor(std::string format, std::array<Py_ssize_t, PyBUF_MAX_NDIM>& shape, std::size_t alignment, std::size_t min_len){
    PyObject* result = PyDict_New();
    PyDict_SetItem(result, PyUnicode_FromString("format"), PyUnicode_FromString(format.c_str()));
    PyDict_SetItem(result, PyUnicode_FromString("shape"), convert_descriptor_item(shape));
    PyDict_SetItem(result, PyUnicode_FromString("alignment"), PyLong_FromLong(alignment));
    PyDict_SetItem(result, PyUnicode_FromString("min_len"), PyLong_FromLong(min_len));
    return result;
}

/* Interface */
struct base_function {
    static const Py_ssize_t queue_variables;
    static const Py_ssize_t user_variables;

};
class simd_function : public base_function, public std::function<void(goopax::resource<T>&,goopax::gpu_uint)> {
public:
    static const Py_ssize_t queue_variables = 3;
    static const Py_ssize_t user_variables = 1;
    static const std::array<Py_ssize_t, PyBUF_MAX_NDIM> Check_shape(goopax::goopax_device& d, const Py_ssize_t len) {
    auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    std::fill(std::begin(shape), std::end(shape), 0);

    /*  Explicit Unrolling  */
    if (d.max_registers() == 0) {
        shape[3] = queue_variables + user_variables;
    } else {
        if (len % (d.max_registers() - queue_variables - user_variables) != 0) {
            shape[3] = 1;
        } else {
            shape[3] = d.max_registers() - queue_variables - user_variables;
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
    static const Py_ssize_t Min_len(goopax_device& d, const Py_ssize_t denominator) {
        Py_ssize_t min_len = lcm(d.default_num_groups_min()*d.default_local_size()*(simd_function::queue_variables+simd_function::user_variables),denominator);//Intel CPU
        if (d.max_registers()>=simd_function::queue_variables+simd_function::user_variables) {
            min_len = lcm(d.default_num_groups_min()*d.default_local_size()*(d.max_registers()-simd_function::queue_variables-simd_function::user_variables),denominator);
        }
        return min_len;
    }
    simd_function(goopax::goopax_device& d) : std::function<void(goopax::resource<T>&,goopax::gpu_uint)>{}, _shape3(d.max_registers() - queue_variables - user_variables) {}
protected:
    unsigned int _shape3;
};

/*  Interface: Types    */
//template<typename FunctionClass> PyObject* host_query(PyObject* empty, PyObject* args);
//template<typename FunctionClass> Py_ssize_t min_len(goopax_device& d, Py_ssize_t denominator);
template<typename FunctionClass> struct BaseFunction{
    PyObject_VAR_HEAD
    goopax_device device;
    FunctionClass* func;
};
void BaseFunction_dealloc(PyObject* self) {
    //printf("Dealloc\n");
    Py_TYPE(self)->tp_free(self);
};
template<typename FunctionClass> PyObject* BaseFunction_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds){
    //printf("New\n");
    BaseFunction<FunctionClass>* obj = (BaseFunction<FunctionClass>*)subtype->tp_alloc(subtype, 0);
    if (obj != NULL) {
        obj->func = (FunctionClass*)malloc(sizeof(FunctionClass));
    }
    return (PyObject*)obj;
};
template<typename FunctionClass> void BaseFunction_free(void *self) {
    //printf("Free\n");
    BaseFunction<FunctionClass>* obj = (BaseFunction<FunctionClass>*)self;
    free((void*)obj->func);
};
template<typename FunctionClass> int BaseFunction_init(PyObject* self, PyObject* args, PyObject* kwds) {
    //printf("Init\n");
    BaseFunction<FunctionClass>* _self = (BaseFunction<FunctionClass>*) self;
    int log_env = 0;
    const unsigned char* gpu_uuid = NULL;
    Py_ssize_t gpu_uuid_length = 0;
    if (PyArg_ParseTuple(args, "iy#", &log_env, &gpu_uuid, &gpu_uuid_length)) {
        const Py_ssize_t d_uuid_length = std::tuple_size<std::array< uint8_t, 16 >>{};
        if (gpu_uuid_length!=d_uuid_length)
            abort();

        envmode env = static_cast<envmode>(1 << log_env);
        std::vector<goopax_device> dvec = devices(static_cast<envmode>(env));
        for (size_t device_i = 0; device_i < dvec.size(); ++device_i) {
            const std::array< uint8_t, d_uuid_length >& d_uuid= dvec[device_i].uuid();
            if (memcmp(gpu_uuid, d_uuid.data(), d_uuid_length) == 0) {
                _self->device = dvec[device_i];
                _self->func = new FunctionClass(dvec[device_i]);
                //printf("Init Success\n");
                return 0;
            }
        }
    }
    _self->func = nullptr;
    return -1;
};
//#include <goopax_extra/output.hpp>
template<typename FunctionClass> PyObject* BaseFunction_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    //printf("Call\n");
    BaseFunction<FunctionClass>* _self = (BaseFunction<FunctionClass>*) self;
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
            //printf("Device min host ptr alignment: %d\n", device_alignment);
            std::size_t padded = (view->len + device_alignment - 1) & ~(device_alignment - 1);
            std::cout << "padded " << padded << " shape0 " << shape0 << " shape1 " << shape1 << " shape2 " << shape2 << " shape3 " << shape3 << " ptr " << view->buf << std::endl;
            buffer<T> buffy(_self->device, view->len/view->itemsize, (T*)view->buf, BUFFER_READ_WRITE);
            kernel<void(buffer<T>&,unsigned int)> process(_self->device, *(_self->func), shape1, shape0*shape1);
            process(buffy,shape2);
            _self->device.wait_all();
            //for(std::size_t i=0; i<shape[0]*shape[1]*shape[2]*shape[3]; ++i)
            //    //printf("%d,",((T*)view->buf)[i]);
            //std::cout<< buffy << std::endl;
            PyBuffer_Release(view);
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
};
/* Optional: Types */
template<typename FunctionClass> PyObject* BaseFunction_host_query(PyObject* empty, PyObject* args, PyObject* kwargs) {
    printf("Host Query\n");
    PY_CXX_CONST char *const kwlist[] = {"format", "length", "denominator", NULL};
    PyObject* Devices = PyList_New(0);
    const char* format = NULL;
    Py_ssize_t format_length = 0;
    Py_ssize_t length = 0;
    Py_ssize_t denominator = 1;
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "s#|nn", kwlist, &format, &format_length, &length, &denominator)) {
        printf("Got format: %s, len: %d, and denominator: %d\n", format, length, denominator);
        for (unsigned int log_env = 0; log_env < 10; ++log_env) {
            envmode env = static_cast<envmode>(1 << log_env);
            std::vector<goopax_device> dvec = test_env(env);
            if (!dvec.empty()){
                for (size_t device_i = 0; device_i < dvec.size(); ++device_i) {
                    //if (!(*shape==gpu_shape && *strides==gpu_strides))
                    Py_ssize_t l = length==0 ? FunctionClass::Min_len(dvec[device_i], denominator) : length;
                    auto registered_type = register_types(dvec[device_i]);
                    for (const auto& dev_ts : registered_type){
                        for (const auto& t : dev_ts.second){
                            if (t.first == format && itemsize(t.first) == itemsize(format)) {
                                PyObject* Device = PyTuple_New(3);
                                PyTuple_SetItem(Device, 0, PyLong_FromLong(log_env));
                                const std::array< uint8_t, 16 > device_id = dvec[device_i].uuid();
                                PyTuple_SetItem(Device, 1, PyBytes_FromStringAndSize(reinterpret_cast<const char*>(&device_id), std::size(device_id)));
                                auto gpu_shape = FunctionClass::Check_shape(dvec[device_i], l);
                                //std::cout << gpu_shape << std::endl;
                                auto gpu_strides = make_strides(gpu_shape, itemsize(format));
                                //std::cout << gpu_strides << std::endl;
                                PyObject* descr = memoryview_descriptor(std::string(format), gpu_shape, dvec[device_i].min_host_ptr_alignment(), FunctionClass::Min_len(dvec[device_i], denominator));
                                PyTuple_SetItem(Device, 2, descr);
                                Py_INCREF(Device);
                                PyList_Append(Devices, Device);
                            }
                        }
                    }
                }
            }
        }
    } else {
        abort();
    }
    return Devices;
}

PyTypeObject BaseFunction_Type(
    const char* BaseFunction_name,
    const char* BaseFunction_doc_string,
    Py_ssize_t basic_size,
    PyMethodDef* methods,
    PyObject* (*python_new_function)(PyTypeObject*, PyObject*, PyObject*),
                               int (*python_init_function)(PyObject*, PyObject*, PyObject*),
                               PyObject* (*python_call_function)(PyObject*, PyObject*, PyObject*),
                               void (*python_free_function)(void*)
) {
    PyTypeObject type = {
        PyVarObject_HEAD_INIT(NULL, 0) // typing.Callable
        BaseFunction_name,
        basic_size,
        0,
        (destructor)BaseFunction_dealloc,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        (ternaryfunc)python_call_function, // BaseFunction_call
        0, 0, 0,
        0,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        PyDoc_STR(BaseFunction_doc_string),
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