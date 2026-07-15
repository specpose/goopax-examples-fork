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
static PyObject* silence_host_query(PyObject* empty, PyObject* args){
    return host_query(SimdCheck_shape<silence>, empty, args);
}