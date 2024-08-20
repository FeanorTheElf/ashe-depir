#pragma once
#include <tuple>

constexpr int F32_MANTISSA_BITS_HALF = 12;
constexpr bool ENABLE_INTERNAL_CHECKS = true;

///
/// Stores a floating-point number by two f32, whose exponents are (at least) 24
/// bits apart. Note that I have not properly thought through cases where either
/// float is subnormal or Inf/NaN.
/// 
/// A lot of the operations are well-known, see e.g. "Implementation of float-float operators on graphics
/// hardware" by Guillaume da Graçca, David Defour (although Knuth's Art also contains some/most of them)
/// 
struct f32x2 {
    float high;
    float low;

    __device__ __host__ inline f32x2() : high(0.f), low(0.f) {}
    __device__ __host__ inline f32x2(float x) : high(x), low(0.f) {}
    __device__ __host__ inline f32x2(const f32x2& x) : high(x.high), low(x.low) {}

    __device__ __host__ inline f32x2(int64_t x) : high(static_cast<float>(x)), low(0.f) {
        low = static_cast<float>(x - static_cast<int64_t>(high));
    }

    __device__ __host__ inline f32x2(double x) : high(static_cast<float>(x)), low(0.f) {
        low = static_cast<float>(x - static_cast<double>(high));
    }
};

template<int S>
__device__ __host__ inline std::tuple<float, float> split(float a) {
    static_assert(S < 24);
    static_assert(S >= 12);
    constexpr float split_factor = static_cast<float>(1 << S) + 1.f;
    float c = split_factor * a;
    float hi = c - (c - a);
    float lo = a - hi;
    return std::make_tuple(hi, lo);
}

__device__ __host__ inline f32x2 add112(float a, float b) {
    f32x2 result;
    result.high = a + b;
    float v = result.high - a;
    result.low = (a - (result.high - v)) + (b - v);
    return result;
}

__device__ __host__ inline f32x2 add222(f32x2 a, f32x2 b) {
    float sum_hi = a.high + b.high;
    if (abs(a.high) < abs(b.high)) {
        std::swap(a, b);
    }
    float s = (((a.high - sum_hi) + b.high) + b.low) + a.low;
    return add112(sum_hi, s);
}

__device__ __host__ inline f32x2 mul112(float a, float b) {
    const auto a_split = split<F32_MANTISSA_BITS_HALF>(a);
    const auto b_split = split<F32_MANTISSA_BITS_HALF>(b);
    return add222(add112(std::get<0>(a_split) * std::get<0>(b_split), std::get<0>(a_split) * std::get<1>(b_split)), add112(std::get<1>(a_split) * std::get<0>(b_split), std::get<1>(a_split) * std::get<1>(b_split)));
}

__device__ __host__ inline f32x2 mul122(f32x2 a, float b) {
    return add222(mul112(a.high, b), mul112(a.low, b));
}

__device__ __host__ inline f32x2 mul222(f32x2 a, f32x2 b) {
    return add222(mul122(a, b.high), mul122(a, b.low));
}

__device__ __host__ inline float round(f32x2 x) {
    return round(x.high) + round(x.low);
}

__device__ __host__ inline f32x2  neg(f32x2 x) {
    x.high = -x.high;
    x.low = -x.low;
    return x;
}