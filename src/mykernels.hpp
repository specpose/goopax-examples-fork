class silence : public simd_function {
public:
    using simd_function::simd_function;
    // Does not work for GOOPAX_ENV=1
    virtual void operator()(goopax::resource<T>& buffy, goopax::gpu_uint shape2) final {
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
};