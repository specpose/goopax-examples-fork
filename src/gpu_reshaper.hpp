#include "gpu_buffer.hpp"
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