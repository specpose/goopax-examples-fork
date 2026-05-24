/*  goopaxinfo  */
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

/*  Goopax  */
bool test_dev(unsigned int env) {
    goopax::goopax_device d = goopax::default_device(static_cast<goopax::envmode>(env));
    if (d.valid())
        return true;
    unsigned int log_env = env >> int(std::log2(env));
    //throw std::runtime_error("Default Device in ENV "+std::to_string(env)+" ("+backend_names[log_env]+")"+" not valid.");
    return false;
}
std::vector<goopax::goopax_device> test_env(unsigned int env) {
    std::vector<goopax::goopax_device> dvec{};
    if (test_dev(env))
        dvec = goopax::devices(static_cast<goopax::envmode>(env));
    return dvec;
}

/*  Logic   */
std::array<Py_ssize_t, PyBUF_MAX_NDIM> check_shape(goopax::goopax_device& d, const Py_ssize_t len) {
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
    if (d.default_global_size_min() % d.default_local_size() != 0 || len / d.default_global_size_min() < 1) {
        shape[0] = 1;
    } else {
        shape[0] = d.default_global_size_min() / d.default_local_size();
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
std::array<Py_ssize_t, PyBUF_MAX_NDIM> make_strides(const std::array<Py_ssize_t, PyBUF_MAX_NDIM>& shape, const Py_ssize_t itemsize) {
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

/*  Python  */
Py_ssize_t itemsize(std::string format){
    PyObject* struct_m = PyImport_ImportModule("struct");
    PyObject* calcsize = PyObject_GetAttrString(struct_m,"calcsize");
    PyObject* calcsize_args = PyTuple_New(1);
    char _python_format[2] = "";
    strcpy(_python_format,format.c_str());
    PyTuple_SetItem(calcsize_args, 0, PyUnicode_FromString(_python_format));
    PyObject* itemsize = (PyObject*)PyObject_CallObject(calcsize, calcsize_args);
    if (PyNumber_Check(itemsize)!=1)
        abort();
    return PyNumber_AsSsize_t(itemsize,NULL);
}
Py_ssize_t ndim(Py_ssize_t * shape) {
    Py_ssize_t ndim = 0;
    for (std::size_t i = 0; i <PyBUF_MAX_NDIM; ++i)
        if (!shape[i]==0)
            ++ndim;
        else
            break;
    return ndim;
};
Py_ssize_t len(Py_ssize_t itemsize, Py_ssize_t * shape) {
    auto len = itemsize;
    for (Py_ssize_t i = 0; i<ndim(shape); ++i)
        len *= shape[i];
    return len;
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