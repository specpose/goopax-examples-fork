/**
   \example goopaxinfo.cpp
   Display goopax device information
 */

#include <goopax>
#include <goopax_extra/output.hpp>
#include <goopax_extra/types.hpp>
#include <iomanip>
#include <Python.h>

using namespace goopax;
using namespace std;

static constexpr const char* backend_names[] = { "CPU", "OpenCL", "CUDA", "Metal", "Vulkan" };

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

std::ostream& operator<<(std::ostream& s, const std::type_info* ti)
{
    if (ti != nullptr)
    {
        s << goopax::pretty_typename(*ti);
    }
    else
    {
        s << "unknown type";
    }
    return s;
}
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
std::array<Py_ssize_t, PyBUF_MAX_NDIM> intel_hd_530(goopax_device& d, const Py_ssize_t len) {
    auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    std::fill(std::begin(shape), std::end(shape), 0);
    shape[3] = 60;
    shape[0] = 768 / 16;
    shape[1] = 16;
    shape[2] = len / (48*16*60);
    return shape;
}

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

const std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> python_types() {
    std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> known_types{};
    known_types[static_cast<type_enum_t>(10 * num_type_modes)].push_back(std::pair(std::string("d"),8));
    known_types[static_cast<type_enum_t>(3 * num_type_modes)].push_back(std::pair(std::string("h"),2));
    return known_types;
};

const std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> register_types(goopax_device& d) {
    const char* delim = " ";
    auto type_to_python = python_types();
    std::map<type_enum_t, std::vector<std::pair<std::string, Py_ssize_t>>> registered_type{};
    for (unsigned int t = 0; t < std::size(type_enum_raw_names); ++t)
    {
        const auto enumerated_type = static_cast<type_enum_t>(t * num_type_modes);
        if (d.support_type(enumerated_type))
        {
            if (type_to_python.find(enumerated_type) != end(type_to_python))
                registered_type[enumerated_type] = type_to_python[enumerated_type];
            cout << delim << type_enum_raw_names[t];
            delim = ", ";
        }
    }
    cout << endl;
    return registered_type;
};

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

int main()
{
    for (unsigned int log_env = 0; log_env < 10; ++log_env)
    {
        goopax::envmode env = static_cast<goopax::envmode>(1 << log_env);
        vector<goopax_device> dvec = test_env(env);
        if (!dvec.empty())
        {
            cout << "\nbackend: " << env;
            if (log_env < std::size(backend_names))
            {
                cout << " (" << backend_names[log_env] << ")";
            }
            cout << endl;
            for (size_t device_i = 0; device_i < dvec.size(); ++device_i)
            {
                auto& d = dvec[device_i];
                cout << "  device " << device_i << "/" << dvec.size() << ":" << endl
                     << "    name=" << d.name() << endl
                     << "    vendor=" << d.vendor() << endl
                     << "    uuid=";
                for (auto& c : d.uuid())
                {
                    cout << hex << std::setfill('0') << std::setw(2) << (int)c << " " << std::dec;
                }
                cout << endl
                     << "    local_size: default=" << d.default_local_size() << ", max=" << d.max_local_size() << endl
                     << "    default global_size: " << d.default_global_size_min() << ".."
                     << d.default_global_size_max() << endl
                     << "    registers: " << d.max_registers() << endl;
                cout << "    supported types:";

                auto registered_type = register_types(d);

                for (const auto& t : registered_type)
                    cout << "\"" << t.second[0].first.c_str() << "\", ";
                cout << endl;

                auto shape = check_shape(d, 92160);
                auto strides = make_strides(shape,std::string("d"),8);
                cout << shape << endl;
                cout << strides << endl;
            }
        }
    }
}
