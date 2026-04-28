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

array<Py_ssize_t, PyBUF_MAX_NDIM> check_shape(goopax_device& d) {
    auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
    fill(std::begin(shape), std::end(shape), 0);
    shape[0] = d.default_global_size_min() % d.default_local_size() == 0 ? d.default_global_size_min() / d.default_local_size() : 0;
    shape[1] = d.default_global_size_min() % d.default_local_size() == 0 ? d.default_local_size() : 0;
    return shape;
};

const map<const char*, vector<string>> populate() {
    map<const char*, vector<string>> known_types{};
    known_types["double"].push_back(string("d"));
    return known_types;
};

const map<const char*, vector<string>> register_types(goopax_device& d) {
    const char* delim = " ";
    auto type_to_python = populate();
    map<const char*, vector<string>> registered_type{};
    for (unsigned int t = 0; t < std::size(type_enum_raw_names); ++t)
    {
        if (d.support_type(static_cast<type_enum_t>(t * num_type_modes)))
        {
            if (type_to_python.find(type_enum_raw_names[t]) != end(type_to_python))
                registered_type[type_enum_raw_names[t]] = type_to_python[type_enum_raw_names[t]];
            cout << delim << type_enum_raw_names[t];
            delim = ", ";
        }
    }
    cout << endl;
    return registered_type;
};

int main()
{
    for (unsigned int log_env = 0; log_env < 10; ++log_env)
    {
        goopax::envmode env = static_cast<goopax::envmode>(1 << log_env);
        vector<goopax_device> dvec = goopax::devices(env);
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

                cout << check_shape(d) << endl;

                for (const auto& t : registered_type)
                    cout << t.first << endl;
            }
        }
    }
}
