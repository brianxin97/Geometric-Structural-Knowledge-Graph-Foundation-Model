#pragma once

#include <limits>

#ifdef __CUDA_ARCH__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

namespace at {

template <class scalar_t>
struct BinaryAdd {
    static constexpr bool is_complex = false;
    HOST_DEVICE static scalar_t forward(const scalar_t* x, const scalar_t* y, int64_t d, int64_t dim) { return x[d] + y[d]; }
    HOST_DEVICE static scalar_t backward_lhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) { return grad_z_d; }
    HOST_DEVICE static scalar_t backward_rhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) { return grad_z_d; }
};

template <class scalar_t>
struct BinaryMul {
    static constexpr bool is_complex = false;
    HOST_DEVICE static scalar_t forward(const scalar_t* x, const scalar_t* y, int64_t d, int64_t dim) { return x[d] * y[d]; }
    HOST_DEVICE static scalar_t backward_lhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) { return grad_z_d * y[d]; }
    HOST_DEVICE static scalar_t backward_rhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) { return grad_z_d * x[d]; }
};

template <class scalar_t>
struct BinaryComplex {
    static constexpr bool is_complex = true;
    HOST_DEVICE static scalar_t forward(const scalar_t* x, const scalar_t* y, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return x[d] * y[d] - x[d + half] * y[d + half];
        else return x[d - half] * y[d] + x[d] * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_lhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * y[d] + grad_z_d_other * y[d + half];
        else return grad_z_d_other * (-y[d]) + grad_z_d * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_rhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * x[d] + grad_z_d_other * x[d + half];
        else return grad_z_d_other * (-x[d]) + grad_z_d * x[d - half];
    }
};

template <class scalar_t>
struct BinarySplitComplex {
    static constexpr bool is_complex = true;
    HOST_DEVICE static scalar_t forward(const scalar_t* x, const scalar_t* y, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return x[d] * y[d] + x[d + half] * y[d + half];
        else return x[d - half] * y[d] + x[d] * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_lhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * y[d] + grad_z_d_other * y[d + half];
        else return grad_z_d_other * y[d] + grad_z_d * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_rhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * x[d] + grad_z_d_other * x[d + half];
        else return grad_z_d_other * x[d] + grad_z_d * x[d - half];
    }
};

template <class scalar_t>
struct BinaryDual {
    static constexpr bool is_complex = true;
    HOST_DEVICE static scalar_t forward(const scalar_t* x, const scalar_t* y, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return x[d] * y[d];
        else return x[d - half] * y[d] + x[d] * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_lhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * y[d] + grad_z_d_other * y[d + half];
        else return grad_z_d * y[d - half];
    }
    HOST_DEVICE static scalar_t backward_rhs(const scalar_t* x, const scalar_t* y, scalar_t grad_z_d, scalar_t grad_z_d_other, int64_t d, int64_t dim) {
        int64_t half = dim / 2;
        if (d < half) return grad_z_d * x[d] + grad_z_d_other * x[d + half];
        else return grad_z_d * x[d - half];
    }
};

template <class scalar_t> struct NaryAdd { HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) { return result + x; } HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) { return 1; } static constexpr scalar_t zero = 0; };
template <class scalar_t> struct NaryMin { HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) { return result < x ? result : x; } HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) { return result == x ? 1 : 0; } static constexpr scalar_t zero = std::numeric_limits<scalar_t>::max(); };
template <class scalar_t> struct NaryMax { HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) { return result > x ? result : x; } HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) { return result == x ? 1 : 0; } static constexpr scalar_t zero = std::numeric_limits<scalar_t>::lowest(); };

} // namespace at