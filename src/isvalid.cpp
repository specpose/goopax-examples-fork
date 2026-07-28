/**
   \example goopaxinfo.cpp
   Display goopax device information
 */

#include <goopax>
#include <goopax_extra/output.hpp>
#include <goopax_extra/types.hpp>
#include <iomanip>

using namespace goopax;
using namespace std;

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

#include <Python.h>
static Py_ssize_t queue_variables = 3;
static Py_ssize_t user_variables = 1;

typedef short T;
#include "gpu_buffer.hpp"
#include "waterfall.hpp"
#include "simd_function.hpp"
#include "mykernels.hpp"

int main()
{
    Py_InitializeEx(0);

    unsigned int numerator = 48000;// sample rate
    unsigned int denominator = 128;// chunk size
    cout << "Assuming denominator " << denominator << endl;
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

                auto registered_type = register_types(d);

                for (const auto& t : registered_type) {
                    cout << "\"" << type_to_python_name(t.first) << "\":\n";
                    for (const auto& s : t.second) {
                        const auto is = itemsize(s.first);
                        auto ml = silence::Min_len(d, denominator);
                        cout << "Assuming minimal aligned buffer size " << ml*is << " bytes" << endl;
                        auto shape = silence::Check_shape(d, ml);
                        cout << shape << endl;
                        auto strides = make_strides(shape, is);
                        cout << strides << endl;
                    }
                    cout << endl;
                }

            }
        }
    }
#if DEBUG
    if (Py_FinalizeEx() < 0)
        abort();
#else
    Py_Finalize();
#endif

}
