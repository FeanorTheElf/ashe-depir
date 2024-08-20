#pragma once
#include <cstdint>
#include <cmath>
#include <assert.h>
#include <string>
#include "f32x2.h"
#include "primes.h"

typedef float ZnEl;

// run test_all_zn() to see that it works for those values; I didn't run the test for larger ones, but just used a conservative bound
template<typename LongFloat>
__device__ __host__ constexpr size_t max_modulus();

template<>
__device__ __host__ constexpr size_t max_modulus<float>() { return 4094; }

template<>
__device__ __host__ constexpr size_t max_modulus<double>() { return 1 << 24; }

template<>
__device__ __host__ constexpr size_t max_modulus<f32x2>() { return 1 << 24; }

template<typename LongFloat>
struct Zn {

private:
    float modulus;
    LongFloat n_inv;

public:
    __device__ __host__ inline Zn(uint32_t modulus) : modulus(static_cast<float>(modulus)), n_inv(1. / static_cast<double>(modulus)) {
        assert(modulus <= max_modulus<LongFloat>());
#ifndef  __CUDA_ARCH__
         if (modulus > max_modulus<LongFloat>()) throw std::invalid_argument("Exceeded maximum allowed modulus");
#endif
    }

    Zn(const Zn&) = default;
    Zn(Zn&&) = default;
    ~Zn() = default;

    Zn& operator=(const Zn&) = default;
    Zn& operator=(Zn&&) = default;

    __device__ __host__ inline int64_t reduce_partial(int64_t value) const;

    __device__ __host__ inline ZnEl reduce_complete(LongFloat value) const;

    __device__ __host__ inline ZnEl mul(ZnEl lhs, ZnEl rhs) const {
        assert(lhs >= -modulus && lhs <= modulus);
        assert(rhs >= -modulus && rhs <= modulus);
        return reduce_complete(LongFloat(lhs) * LongFloat(rhs));
    }

    __device__ __host__ inline float add(ZnEl lhs, ZnEl rhs) const {
        assert(lhs >= -modulus && lhs <= modulus);
        assert(rhs >= -modulus && rhs <= modulus);
        float result = lhs + rhs;
        if (result < -modulus) {
            result += modulus;
        }
        else if (result > modulus) {
            result -= modulus;
        }
        return result;
    }

    __device__ __host__ inline ZnEl from_int(int32_t n) const {
        int64_t result = n;
        result = reduce_partial(result);
        return reduce_complete(LongFloat(result));
    }

    __host__ inline ZnEl from_int(int64_t n) const {
        int64_t current = n;
        current = reduce_partial(current);
        current = reduce_partial(current);
        LongFloat result = reduce_complete(LongFloat(current));
        return result;
    }

    __device__ __host__ inline ZnEl mul_add(ZnEl lhs, ZnEl rhs, ZnEl add) const {
        assert(lhs >= -modulus && lhs <= modulus);
        assert(rhs >= -modulus && rhs <= modulus);
        assert(add >= -modulus && add <= modulus);
        return reduce_complete(LongFloat(lhs) * LongFloat(rhs) + LongFloat(add));
    }

    __device__ __host__ inline uint32_t ring_modulus() const {
        return static_cast<uint32_t>(modulus);
    }

    __device__ __host__ inline uint32_t lift(ZnEl x) const {
        uint32_t result = static_cast<uint32_t>(x + modulus);
        if (result >= modulus) {
            result -= static_cast<uint32_t>(modulus);
        }
        if (result >= modulus) {
            result -= static_cast<uint32_t>(modulus);
        }
        return result;
    }
};

template<typename LongFloat>
inline void test_zn(uint32_t modulus) {
    const int64_t n = static_cast<int64_t>(modulus);
    const Zn<LongFloat> ring{ modulus };
    const int64_t max_value = n * n + n;
    for (int64_t a = -max_value; a <= max_value; ++a) {
        const int64_t expected = a % n;
        const int64_t actual = static_cast<int64_t>(ring.reduce_complete(a));

        if ((actual - expected) % n != 0) {
            throw "failed congruence";
        }
        else if (actual < -n || actual > n) {
            throw "failed reducedness";
        }
    }
}

template<typename LongFloat>
inline void test_all_zn() {
    for (uint32_t n = 10000; ; ++n) {
        try {
            test_zn<LongFloat>(n);
            std::cout << n << std::endl;
        }
        catch (const char* exception) {
            throw std::string("failed for ") + std::to_string(n) + "; " + exception;
        }
    }
}

template<>
inline __device__ __host__ int64_t Zn<float>::reduce_partial(int64_t value) const
{
    return value - static_cast<int64_t>(modulus) * static_cast<int64_t>(round(static_cast<float>(value) * n_inv));
}

template<>
inline __device__ __host__ ZnEl Zn<float>::reduce_complete(float value) const
{
    return value - modulus * round(value * n_inv);
}

template<>
inline __device__ __host__ int64_t Zn<double>::reduce_partial(int64_t value) const
{
    return value - static_cast<int64_t>(modulus) * static_cast<int64_t>(round(static_cast<float>(value) * static_cast<float>(n_inv)));
}

template<>
inline __device__ __host__ ZnEl Zn<double>::reduce_complete(double value) const
{
    return static_cast<float>(value - static_cast<double>(modulus) * round(value * n_inv));
}

template<>
inline __device__ __host__ int64_t Zn<f32x2>::reduce_partial(int64_t value) const
{
    return value - static_cast<int64_t>(modulus) * static_cast<int64_t>(round(static_cast<float>(value) * n_inv.high));
}

template<>
inline __device__ __host__ ZnEl Zn<f32x2>::reduce_complete(f32x2 value) const
{
    return round(add222(value, neg(mul112(modulus, round(mul122(value, n_inv.high))))));
}
