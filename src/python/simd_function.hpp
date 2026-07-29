class simd_function : public base_function
{
public:
    static const Py_ssize_t queue_variables = 3;
    static const Py_ssize_t user_variables = 1;
    static const std::array<Py_ssize_t, PyBUF_MAX_NDIM> Check_shape(goopax::goopax_device& d, const Py_ssize_t len)
    {
        auto shape = std::array<Py_ssize_t, PyBUF_MAX_NDIM>{};
        std::fill(std::begin(shape), std::end(shape), 0);

        /*  Explicit Unrolling  */
        if (d.max_registers() == 0)
        {
            shape[3] = queue_variables + user_variables;
        }
        else
        {
            if (len % (d.max_registers() - queue_variables - user_variables) != 0)
            {
                shape[3] = 1;
            }
            else
            {
                shape[3] = d.max_registers() - queue_variables - user_variables;
            }
        }

        /*  Workgroups  */
        if (len / d.default_num_groups_min() < 1)
        {
            shape[0] = 1;
        }
        else
        {
            shape[0] = d.default_num_groups_min();
        }

        /*  ALUs  */
        if (len / d.default_local_size() < 1)
        {
            shape[1] = 1;
        }
        else
        {
            shape[1] = d.default_local_size();
        }

        /*  Queue  */
        if (len / shape[0] / shape[1] / shape[3] < 1)
        {
            shape[2] = 1;
        }
        else
        {
            shape[2] = len / (shape[0] * shape[1] * shape[3]);
        }

        return shape;
    };
    static const Py_ssize_t Min_len(goopax_device& d, const Py_ssize_t denominator)
    {
        Py_ssize_t min_len = std::lcm(d.default_num_groups_min() * d.default_local_size()
                                          * (simd_function::queue_variables + simd_function::user_variables),
                                      denominator); // Intel CPU
        if (d.max_registers() >= simd_function::queue_variables + simd_function::user_variables)
        {
            min_len =
                std::lcm(d.default_num_groups_min() * d.default_local_size()
                             * (d.max_registers() - simd_function::queue_variables - simd_function::user_variables),
                         denominator);
        }
        return min_len;
    }
    simd_function(goopax::goopax_device& d)
        : base_function()
        , _shape3(d.max_registers() - queue_variables - user_variables)
    {
    }

protected:
    unsigned int _shape3;
};