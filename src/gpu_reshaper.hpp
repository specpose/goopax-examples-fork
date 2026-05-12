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
    goopax_device d = goopax::default_device(static_cast<goopax::envmode>(env));
    if (d.valid())
        return true;
    unsigned int log_env = env >> int(std::log2(env));
    //throw std::runtime_error("Default Device in ENV "+std::to_string(env)+" ("+backend_names[log_env]+")"+" not valid.");
    return false;
}
std::vector<goopax_device> test_env(unsigned int env) {
    std::vector<goopax_device> dvec{};
    if (test_dev(env))
        dvec = goopax::devices(static_cast<goopax::envmode>(env));
    return dvec;
}

/*  Logic   */
std::array<Py_ssize_t, PyBUF_MAX_NDIM> check_shape(goopax_device& d, const Py_ssize_t len) {
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