/**
   \example goopaxinfo.cpp
   Display goopax device information
 */

#include <goopax>
#include <goopax_extra/output.hpp>
#include <goopax_extra/types.hpp>
#include <Python.h>
static Py_ssize_t queue_variables = 3;
static Py_ssize_t user_variables = 1;
#include "gpu_reshaper.hpp"

using namespace goopax;
using namespace std;

#include <map>
namespace std
{
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

#include <iomanip>
int main()
{
    Py_InitializeEx(0);

    unsigned int bin_size = 2400;//48000
    cout << "Assuming bin size " << bin_size << endl;
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
                        const auto _itemsize = itemsize(s.first);
                        unsigned int min_len = lcm((d.default_global_size_min()/d.default_local_size())*d.default_local_size()*(queue_variables+user_variables)*_itemsize,d.min_host_ptr_alignment());//Intel CPU
                        if (d.max_registers()>=queue_variables+user_variables) {
                            min_len = lcm((d.default_global_size_min()/d.default_local_size())*d.default_local_size()*(d.max_registers()-queue_variables-user_variables)*_itemsize,d.min_host_ptr_alignment());
                        }
                        cout << "Assuming minimal aligned buffer size " << min_len << " bytes" << endl;
                        auto shape = check_shape(d, min_len);
                        cout << shape << endl;
                        auto strides = make_strides(shape, _itemsize);
                        cout << strides << endl;
                        auto new_bin_size = lcm(shape[2]*shape[3],bin_size);
                        cout << "Assuming minimal bin length " << new_bin_size << " symbols, actual bin length " << shape[2]*shape[3] << " symbols" << endl;
                        auto queue_factor = new_bin_size/(shape[2]*shape[3]);
                        cout << "Adjusted minimal queue length " << queue_factor*shape[2] << " symbols, actual queue length " << shape[2] << " symbols" << endl;
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
