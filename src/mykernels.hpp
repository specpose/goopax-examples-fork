class silence : public std::function<void(goopax::resource<T>&,goopax::gpu_uint)> {
public:
    static const Py_ssize_t queue_variables = 3;
    static const Py_ssize_t user_variables = 1;
    silence(goopax::goopax_device& d) : std::function<void(goopax::resource<T>&,goopax::gpu_uint)>{}, _shape3(d.max_registers() - queue_variables - user_variables) {}
    // Does not work for GOOPAX_ENV=1
    void operator()(goopax::resource<T>& buffy, goopax::gpu_uint shape2){
        //gpu_for(0+global_id()*_shape3, buffy.size(), global_size()*_shape3, [&buffy,this](gpu_uint k){
        //    for(unsigned int l = 0; l < _shape3; ++l) {
        //        buffy[k+l] -= 257; // T
        //    }
        //});
        gpu_for_group(0, num_groups(), [&buffy,shape2,this](gpu_uint i){
            gpu_for_local(0, local_size(), [&buffy,shape2,this](gpu_uint j){
                gpu_for(0, shape2, [&buffy,shape2,this](gpu_uint k){
                    for (unsigned int l = 0; l < _shape3; ++l) {
                        buffy[global_id()*shape2*_shape3+k*_shape3+l] -= 257; // T
                    }
                });
            });
        });
    };
private:
    unsigned int _shape3;
};
/*
struct silence {
    silence(goopax_device& d) : _shape3(d.max_registers() - queue_variables - user_variables) {}
    unsigned int _shape3;
    static const unsigned int queue_variables = 3;
    static const unsigned int user_variables = 1;
    std::function<void(resource<T>&,goopax::gpu_uint)> afunc = [&](resource<T>& buffy, goopax::gpu_uint shape2) {};
};
*/
template<> PyObject* host_query<silence>(PyObject* empty, PyObject* args){
    return host_query(SimdCheck_shape<silence>, empty, args);
}
template<> Py_ssize_t min_len<silence>(goopax_device& d, Py_ssize_t denominator) {
    Py_ssize_t min_len = lcm(d.default_num_groups_min()*d.default_local_size()*(silence::queue_variables+silence::user_variables),denominator);//Intel CPU
    if (d.max_registers()>=silence::queue_variables+silence::user_variables) {
        min_len = lcm(d.default_num_groups_min()*d.default_local_size()*(d.max_registers()-silence::queue_variables-silence::user_variables),denominator);
    }
    return min_len;
}