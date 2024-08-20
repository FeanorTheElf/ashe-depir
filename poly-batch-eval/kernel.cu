
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <fstream>
#include <stdio.h>
#include <cstddef>
#include <iostream>
#include <assert.h>
#include <numeric>
#include <optional>
#include <chrono>
#include <vector>
#include <chrono>

#include "zn.h"

#define cuda_check(ans) { cuda_error_check((ans), __FILE__, __LINE__); }

inline void cuda_error_check(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << "  at " << file << ":" << line << std::endl;
        if (abort) {
            exit(code);
        }
    }
}

typedef unsigned char byte;

__device__ __host__ inline size_t binomial(size_t n, size_t k) {
    assert(k <= n);
    k = std::min(k, n - k);
    size_t num = 1;
    size_t den = 1;
    for (size_t i = 0; i < k; ++i) {
        num *= (n - i);
        den *= (i + 1);
    }
    return num / den;
}

template<size_t k>
__device__ __host__ constexpr size_t static_binomial(size_t n) {
    if (k > n) {
        return 0;
    }
    size_t num = 1;
    size_t den = 1;
    for (size_t i = 0; i < k; ++i) {
        num *= (n - i);
        den *= (i + 1);
    }
    return num / den;
}

template<size_t k>
__device__ __host__ constexpr size_t static_pow(size_t a) {
    size_t result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= a;
    }
    return result;
}

template<>
__device__ __host__ constexpr size_t static_binomial<0>(size_t n) {
    return 1;
}

template<size_t m>
struct DegLexBlockIter {

private:
    size_t d;
    size_t current_degree;
    size_t current_power;
    size_t current_start;
    size_t current_end;

    __device__ __host__ constexpr inline DegLexBlockIter(size_t d, size_t current_power, size_t current_degree, size_t current_start, size_t current_end) : d(d), current_power(current_power), current_degree(current_degree), current_start(current_start), current_end(current_end) {}

public:

    DegLexBlockIter(const DegLexBlockIter&) = default;
    DegLexBlockIter(DegLexBlockIter&&) = default;
    ~DegLexBlockIter() = default;

    DegLexBlockIter& operator=(const DegLexBlockIter&) = default;
    DegLexBlockIter& operator=(DegLexBlockIter&&) = default;

    __device__ __host__ inline bool operator!=(const DegLexBlockIter& other) const {
        return current_power != other.current_power || current_degree != other.current_degree;
    }

    __device__ __host__ inline DegLexBlockIter& operator++() {
        static_assert(m >= 1, "m must not be zero");
        if constexpr (m == 1) {
            current_start += 1;
            current_end += 1;
            current_power += 1;
            current_degree += 1;
        }
        else {
            current_power += 1;
            if (current_power <= current_degree) {
                current_start = current_end;
                current_end = current_end + static_binomial<m - 2>(current_degree + m - current_power - 2);
            }
            else {
                assert(current_power == current_degree + 1);
                current_power = 0;
                current_degree += 1;
                current_start = current_end;
                current_end = current_end + static_binomial<m - 2>(current_degree + m - current_power - 2);
            }
        }
        return *this;
    }

    __device__ __host__ inline size_t power() const {
        return current_power;
    }

    __device__ __host__ inline size_t degree() const {
        return current_degree;
    }

    __device__ __host__ inline size_t range_start() const {
        return current_start;
    }

    __device__ __host__ inline size_t range_end() const {
        return current_end;
    }

    __device__ __host__ static inline DegLexBlockIter begin(size_t d) {
        return DegLexBlockIter(d, 0, 0, 0, 1);
    }

    __device__ __host__ static constexpr inline DegLexBlockIter end(size_t d) {
        return DegLexBlockIter(d, 0, d + 1, static_binomial<m>(d + m), static_binomial<m>(d + m));
    }
};

template<typename CG, typename LongFloat, size_t m>
__device__ __host__ inline void specialize_poly_last(const CG& cg, const Zn<LongFloat>& ring, const ZnEl* poly, ZnEl* output, ZnEl value, const size_t d) {
    static_assert(m > 1, "If m == 1, please use evaluate_poly()");

    const size_t cg_offset = cg.thread_rank();
    const size_t cg_step = cg.size();

    for (size_t i = cg_offset; i < static_binomial<m - 1>(d + m - 1); i += cg_step) {
        output[i] = 0.;
    }

    cg.sync();

    const DegLexBlockIter<m> end = DegLexBlockIter<m>::end(d);
    DegLexBlockIter<m> it = DegLexBlockIter<m>::begin(d);
    size_t out_index = 0;
    size_t out_deg = 0;
    ZnEl factor;
    for (; it != end; ++it) {
        if (it.power() == 0) {
            factor = 1.;
        }
        if (it.degree() - it.power() != out_deg) {
            assert(it.power() == 0);
            out_deg = it.degree();
            out_index = 0;
            for (size_t j = 0; j < out_deg; ++j) {
                out_index += static_binomial<m - 2>(j + m - 2);
            }
        }
        assert(out_deg == it.degree() - it.power());

        for (size_t i = it.range_start() + cg_offset; i < it.range_end(); i += cg_step) {
            output[i - it.range_start() + out_index] = ring.mul_add(poly[i], factor, output[i - it.range_start() + out_index]);
        }

        cg.sync();

        factor = ring.mul(factor, value);
        out_deg -= 1;
        out_index -= static_binomial<m - 2>(out_deg + m - 2);
    }
}

template<typename LongFloat>
__device__ __host__ inline ZnEl evaluate_poly(const Zn<LongFloat>& ring, const ZnEl* poly, ZnEl value, const size_t d) {
    ZnEl result = 0;
    for (int i = d; i >= 0; --i) {
        result = ring.mul_add(result, value, poly[i]);
    }
    return result;
}

template<typename LongFloat, size_t m>
__device__ __host__ inline void index_to_point(const Zn<LongFloat>& ring, int index, ZnEl* output, const size_t n) {
    const int16_t start_end = static_cast<int16_t>(n / 2);
    for (int i = 0; i < m; ++i) {
        const int16_t current = static_cast<int16_t>(index % n);
        output[i] = ring.from_int((int16_t)(current - start_end));
        index = (index - current) / n;
    }
    assert(index == 0);
}

__device__ __host__ void swap(ZnEl*& lhs, ZnEl*& rhs) {
    ZnEl* tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

template<typename CG, typename LongFloat, size_t m>
__device__ __host__ inline void specialize_poly(const CG& cg, const Zn<LongFloat>& ring, ZnEl* __restrict__ poly, const size_t d, const ZnEl* __restrict__ point, const ZnEl* __restrict__ last_point, bool always_specialize) {
    static_assert(m >= 1, "Wrong implementation of specialization");
    if constexpr (m > 1) {
        if (always_specialize || (point[m - 1] != last_point[m - 1])) {
            specialize_poly_last<CG, LongFloat, m>(cg, ring, poly, &poly[static_binomial<m>(d + m)], point[m - 1], d);
            specialize_poly<CG, LongFloat, m - 1>(cg, ring, &poly[static_binomial<m>(d + m)], d, point, last_point, true);
        }
        else {
            specialize_poly<CG, LongFloat, m - 1>(cg, ring, &poly[static_binomial<m>(d + m)], d, point, last_point, false);
        }
    }
}

template<size_t m>
__device__ __host__ inline ZnEl* select_specialized_poly(ZnEl* poly, const size_t d) {
    static_assert(m >= 1, "Wrong implementation of specialization");
    if constexpr (m > 1) {
        return select_specialized_poly<m - 1>(&poly[static_binomial<m>(d + m)], d);
    }
    else {
        return poly;
    }
}

template<size_t m>
__device__ __host__ inline size_t intermediate_size(const size_t d) {
    static_assert(m >= 1, "Wrong implementation of specialization");
    if constexpr (m > 1) {
        return intermediate_size<m - 1>(d) + static_binomial<m>(d + m);
    }
    else {
        return static_binomial<m>(d + m);
    }
}

template<typename CG, typename LongFloat, size_t m>
__device__ __host__ inline void evaluate_poly_grid(
    const CG& cg, 
    const Zn<LongFloat>& ring,
    const ZnEl* const* const polys, 
    const size_t poly_count, 
    ZnEl* const* const intermediate_global,
    ZnEl* point, 
    ZnEl* last_point, 
    ZnEl* const output, 
    const size_t start_index, 
    const size_t end_index, 
    const size_t d, 
    const size_t point_step, 
    const size_t n
) {
    assert(start_index % n == 0);
    assert(end_index % n == 0);

    const size_t cg_offset = cg.thread_rank();
    const size_t cg_step = cg.size();

    for (size_t i = 1; i < m; ++i) {
        last_point[i] = INFINITY;
    }

    for (size_t i = start_index / n; i < end_index / n; i += point_step) {

        index_to_point<LongFloat, m - 1>(ring, i, &point[1], n);
        assert(point[0] == 0.);

        for (size_t j = 0; j < poly_count; ++j) {

            const size_t degree = d - j;
            ZnEl* intermediate = &intermediate_global[j][intermediate_size<m - 1>(degree) * cg.group_index().x];

            cg.sync();

            if (last_point[m - 1] != point[m - 1]) {
                specialize_poly_last<CG, LongFloat, m>(cg, ring, polys[j], intermediate, point[m - 1], degree);
            }
            specialize_poly<CG, LongFloat, m - 1>(cg, ring, intermediate, degree, point, last_point, last_point[m - 1] != point[m - 1]);

            ZnEl* const specialized_poly = select_specialized_poly<m - 1>(intermediate, degree);

            cg.sync();

            const int start_end = static_cast<int32_t>(n / 2);
            for (int32_t last_coord = cg_offset - start_end; last_coord <= start_end; last_coord += cg_step) {
                const size_t out_index = (i * n - start_index + last_coord + start_end) * poly_count + j;
                output[out_index] = evaluate_poly<LongFloat>(ring, specialized_poly, ring.from_int(last_coord), degree);
            }

            cg.sync();
        }

        swap(point, last_point);
    }
}

struct SingleThreadCG {

    __device__ __host__ void sync() const {}

    __device__ __host__ size_t thread_rank() const {
        return 0;
    }

    __device__ __host__ size_t size() const {
        return 1;
    }

    __device__ __host__ dim3 group_index() const {
        return dim3(0, 0, 0);
    }
};

template<typename LongFloat, size_t m>
__global__ void evaluate_poly_kernel(Zn<LongFloat> ring, const ZnEl* const* const polys, const size_t polys_count, ZnEl* const output, ZnEl* const* const intermediate, const size_t start, const size_t end, const size_t d, const size_t n) {

    ZnEl point_and_last_point[m * 2];

    assert(start % n == 0);
    assert(end % n == 0);

    cooperative_groups::thread_block cg = cooperative_groups::this_thread_block();

    const size_t block_offset = blockIdx.x * n;
    const size_t block_size = gridDim.x;

    evaluate_poly_grid<cooperative_groups::thread_block, LongFloat, m>(
        cg, 
        ring, 
        polys,
        polys_count,
        intermediate,
        point_and_last_point, 
        &point_and_last_point[m],
        &output[block_offset * polys_count],
        start + block_offset,
        end, 
        d,
        block_size,
        n
    );
}

template<typename LongFloat, size_t m>
std::vector<ZnEl> evaluate_poly_parallel(Zn<LongFloat> ring, const std::vector<std::vector<ZnEl>>& polys, const size_t start, const size_t end, const size_t d, const size_t n) {
    assert(start % n == 0);
    assert(end % n == 0);
    assert(start < end);
    assert(end <= static_pow<m>(n));
    for (size_t i = 0; i < polys.size(); ++i) {
        assert(polys[i].size() == static_binomial<m>(d + m - i));
    }

    const size_t block_count = 2048;
    const size_t block_size = 64;

    std::vector<ZnEl*> polys_device;
    for (size_t i = 0; i < polys.size(); ++i) {
        ZnEl* poly_device;
        cuda_check(cudaMalloc(&poly_device, polys[i].size() * sizeof(ZnEl)));
        polys_device.push_back(poly_device);
        if (polys[i].size() > 0) {
            cuda_check(cudaMemcpy(poly_device, &polys[i][0], polys[i].size() * sizeof(ZnEl), cudaMemcpyHostToDevice));
        }
    }

    const ZnEl** polys_list_device;
    cuda_check(cudaMalloc(&polys_list_device, polys_device.size() * sizeof(ZnEl*)));
    cuda_check(cudaMemcpy(polys_list_device, &polys_device[0], polys_device.size() * sizeof(ZnEl*), cudaMemcpyHostToDevice));

    std::vector<ZnEl*> intermediates_device;
    for (size_t i = 0; i < polys.size(); ++i) {
        ZnEl* intermediate_device;
        cuda_check(cudaMalloc(&intermediate_device, block_count * intermediate_size<m - 1>(d - i) * sizeof(ZnEl)));
        intermediates_device.push_back(intermediate_device);
    }

    ZnEl** intermediates_list_device;
    cuda_check(cudaMalloc(&intermediates_list_device, intermediates_device.size() * sizeof(ZnEl*)));
    cuda_check(cudaMemcpy(intermediates_list_device, &intermediates_device[0], intermediates_device.size() * sizeof(ZnEl*), cudaMemcpyHostToDevice));

    ZnEl* result_device;
    cuda_check(cudaMalloc(&result_device, (end - start) * polys.size() * sizeof(ZnEl)));

    evaluate_poly_kernel <LongFloat, m> << <block_count, block_size>> > (ring, polys_list_device, polys.size(), result_device, intermediates_list_device, start, end, d, n);
    cuda_check(cudaDeviceSynchronize());

    std::vector<ZnEl> result;
    result.resize((end - start) * polys.size());
    cuda_check(cudaMemcpy(&result[0], result_device, (end - start) * polys.size() * sizeof(ZnEl), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < polys.size(); ++i) {
        cuda_check(cudaFree(polys_device[i]));
    }
    cuda_check(cudaFree(polys_list_device));
    for (size_t i = 0; i < intermediates_device.size(); ++i) {
        cuda_check(cudaFree(intermediates_device[i]));
    }
    cuda_check(cudaFree(intermediates_list_device));
    cuda_check(cudaFree(result_device));

    return result;
}

template<typename LongFloat>
std::vector<ZnEl> read_polynomial(const Zn<LongFloat>& ring, const char* filename, const size_t d, const size_t m) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: \"" << filename << "\": " << std::strerror(errno) << std::endl;
        throw std::exception();
    }
    std::vector<ZnEl> result;
    result.reserve(binomial(d + m, m));

    typedef int64_t InputInt;

    constexpr size_t read_count = 256;
    // native bit ordering is little endian, so this should work
    InputInt read_buffer[read_count];
    for (size_t i = 0; i < (binomial(d + m, m) - 1) / read_count + 1; ++i) {
        const size_t read_elements = std::min(read_count, binomial(d + m, m) - i * read_count);
        file.read(reinterpret_cast<char*>(&read_buffer), read_elements * sizeof(InputInt));
        if (file.eof() || file.fail()) {
            std::cerr << "Error: \"" << filename << "\" is too short" << std::endl;
            throw std::exception();
        }
        for (size_t j = 0; j < read_elements; ++j) {
            result.push_back(ring.from_int(static_cast<int64_t>(read_buffer[j])));
        }
    }
    file.read(reinterpret_cast<char*>(&read_buffer), sizeof(InputInt));
    if (!file.eof()) {
        std::cerr << "Error: \"" << filename << "\" is too long" << std::endl;
        throw std::exception();
    }
    file.close();
    assert(result.size() == binomial(d + m, m));
    return result;
}

template<typename LongFloat>
void write_evaluations(const LongFloat& ring, const char* filename, const std::vector<ZnEl>& out_data) {
    std::ofstream file(filename, std::ios::binary | std::ios::out | std::ios::app);
    if (!file) {
        std::cerr << "Error: \"" << filename << "\": " << std::strerror(errno) << std::endl;
        throw std::exception();
    }

    typedef uint16_t OutputUnsignedInt;

    assert(ring.ring_modulus() - 1 <= std::numeric_limits<OutputUnsignedInt>().max());
    constexpr size_t write_count = 256;
    OutputUnsignedInt write_buffer[write_count];
    for (size_t i = 0; i < (out_data.size() - 1) / write_count + 1; ++i) {
        for (size_t j = 0; j < std::min(write_count, out_data.size() - i * write_count); ++j) {
            write_buffer[j] = ring.lift(out_data[i * write_count + j]);
        }
        file.write(reinterpret_cast<const char*>(&write_buffer), sizeof(OutputUnsignedInt) * std::min(write_count, out_data.size() - i * write_count));
    }
    file.close();
}

void test_degrevlex_block_iter() {
    DegLexBlockIter<3> it = DegLexBlockIter<3>::begin(2);
    assert(0 == it.degree());
    assert(0 == it.power());
    assert(0 == it.range_start());
    assert(1 == it.range_end());
    ++it;
    assert(it != DegLexBlockIter<3>::end(2));
    assert(1 == it.degree());
    assert(0 == it.power());
    assert(1 == it.range_start());
    assert(3 == it.range_end());
    ++it;
    assert(it != DegLexBlockIter<3>::end(2));
    assert(1 == it.degree());
    assert(1 == it.power());
    assert(3 == it.range_start());
    assert(4 == it.range_end());
    ++it;
    assert(it != DegLexBlockIter<3>::end(2));
    assert(2 == it.degree());
    assert(0 == it.power());
    assert(4 == it.range_start());
    assert(7 == it.range_end());
    ++it;
    assert(it != DegLexBlockIter<3>::end(2));
    assert(2 == it.degree());
    assert(1 == it.power());
    assert(7 == it.range_start());
    assert(9 == it.range_end());
    ++it;
    assert(it != DegLexBlockIter<3>::end(2));
    assert(2 == it.degree());
    assert(2 == it.power());
    assert(9 == it.range_start());
    assert(10 == it.range_end());
    ++it;
    assert(!(it != DegLexBlockIter<3>::end(2)));

    std::cout << "test_degrevlex_block_iter() success" << std::endl;
}

void test_evaluate_poly_grid() {
    Zn<float> ring{ 17 };
    const size_t d = 2;
    constexpr size_t m = 3;

    // 1 - y + x^2 + yx + 2zx + 3zy + z^2
    ZnEl poly[10] = { 1, 0, -1, 0, 1, 1, 0, 2, 3, 1 };

    // temporary memory
    ZnEl intermediate[6 + 3] = {};
    ZnEl out[17 * 17 * 17] = {};
    ZnEl point[3] = {};
    ZnEl last_point[3] = {};

    const ZnEl* poly_ptr = poly;
    ZnEl* const intermediate_ptr = intermediate;
    evaluate_poly_grid<SingleThreadCG, float, m>(SingleThreadCG{}, ring, & poly_ptr, 1, &intermediate_ptr, point, last_point, out, 0, 17 * 17 * 17, d, 1, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (1 + x * x - y + y * x + 2 * z * x + 3 * z * y + z * z + 17 * 17 * 17 * 17) % 17);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid() success" << std::endl;
}

void test_evaluate_poly_grid_order() {
    Zn<float> ring{ 17 };
    const size_t d = 2;
    constexpr size_t m = 3;

    // x^2 + 2xy + 3y^2 + 4zx + 5zy + 6z^2
    ZnEl poly[10] = { 0, 0, 0, 0, 1, 2, 3, 4, 5, 6 };

    // temporary memory
    ZnEl intermediate[6 + 3] = {};
    ZnEl out[17 * 17 * 17] = {};
    ZnEl point[3] = {};
    ZnEl last_point[3] = {};

    const ZnEl* poly_ptr = poly;
    ZnEl* const intermediate_ptr = intermediate;
    evaluate_poly_grid<SingleThreadCG, float, m>(SingleThreadCG{}, ring, &poly_ptr, 1, &intermediate_ptr, point, last_point, out, 0, 17 * 17 * 17, d, 1, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (x * x + 2 * x * y + 3 * y * y + 4 * z * x + 5 * z * y + 6 * z * z + 17 * 17 * 17 * 17) % 17);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid() success" << std::endl;
}

void test_evaluate_poly_grid_larger() {
    Zn<float> ring{ 17 };
    const size_t d = 3;
    constexpr size_t m = 3;

    // 1 + x^2 + y^2 + 2xz + 3z^3
    ZnEl poly[20] = { 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 };

    // temporary memory
    ZnEl intermediate[10 + 4] = {};
    ZnEl out[17 * 17 * 17] = {};
    ZnEl point[3] = {};
    ZnEl last_point[3] = {};

    const ZnEl* poly_ptr = poly;
    ZnEl* const intermediate_ptr = intermediate;
    evaluate_poly_grid<SingleThreadCG, float, m>(SingleThreadCG{}, ring, &poly_ptr, 1, &intermediate_ptr, point, last_point, out, 0, 17 * 17 * 17, d, 1, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (1 + x * x + y * y + 2 * x * z + 3 * z * z * z + 17 * 17 * 17 * 17) % 17);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_larger() success" << std::endl;
}

void test_evaluate_poly_grid_four_vars() {
    Zn<float> ring{ 17 };
    const size_t d = 2;
    constexpr size_t m = 4;

    // 3 + 3xy + xz - 2zw - 7w
    ZnEl poly[15] = { 3, 0, 0, 0, -7, 0, 3, 0, 1, 0, 0, 0, 0, -2, 0 };

    // temporary memory
    ZnEl intermediate[10 + 6 + 3] = {};
    ZnEl out[17 * 17 * 17 * 17] = {};
    ZnEl point[4] = {};
    ZnEl last_point[4] = {};

    const ZnEl* poly_ptr = poly;
    ZnEl* const intermediate_ptr = intermediate;
    evaluate_poly_grid<SingleThreadCG, float, m>(SingleThreadCG{}, ring, &poly_ptr, 1, &intermediate_ptr, point, last_point, out, 0, 17 * 17 * 17 * 17, d, 1, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                for (int w = -8;  w < 9; ++w) {
                    assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17 + (w + 8) * 17 * 17 * 17]) == (3 * x * y + x * z - 2 * z * w - 7 * w + 3 + 17 * 17 * 17 * 17) % 17);
                }
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_four_vars() success" << std::endl;
}

void test_evaluate_poly_grid_larger_p_squared() {
    Zn<float> ring{ 17 * 17 };
    const size_t d = 3;
    constexpr size_t m = 3;

    // 1 + x^2 + y^2 + 2xz + 3z^3
    ZnEl poly[20] = { 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 };

    // temporary memory
    ZnEl intermediate[10 + 4] = {};
    ZnEl out[17 * 17 * 17] = {};
    ZnEl point[3] = {};
    ZnEl last_point[3] = {};

    const ZnEl* poly_ptr = poly;
    ZnEl* const intermediate_ptr = intermediate;
    evaluate_poly_grid<SingleThreadCG, float, m>(SingleThreadCG{}, ring, &poly_ptr, 1, &intermediate_ptr, point, last_point, out, 0, 17 * 17 * 17, d, 1, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (1 + x * x + y * y + 2 * x * z + 3 * z * z * z + 17 * 17 * 17 * 17) % (17 * 17));
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_larger_p_squared() success" << std::endl;
}

void test_evaluate_poly_grid_device() {
    Zn<float> ring{ 17 };
    const size_t d = 3;
    constexpr size_t m = 3;

    // 1 + x^2 + y^2 + 2xz + 3z^3
    std::vector<ZnEl> poly = { 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 };

    std::vector<ZnEl> out = evaluate_poly_parallel<float, m>(ring, { poly }, 0, 17 * 17 * 17, d, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (1 + x * x + y * y + 2 * x * z + 3 * z * z * z + 17 * 17 * 17 * 17) % 17);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_device() success" << std::endl;
}

void test_evaluate_poly_grid_device_second_half() {
    Zn<float> ring{ 17 };
    const size_t d = 3;
    constexpr size_t m = 3;

    // 1 + x^2 + y^2 + 2xz + 3z^3
    std::vector<ZnEl> poly = { 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 };

    std::vector<ZnEl> out = evaluate_poly_parallel<float, m>(ring, { poly }, 8 * 17 * 17, 17 * 17 * 17, d, 17);

    for (int z = 0; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + z * 17 * 17]) == (1 + x * x + y * y + 2 * x * z + 3 * z * z * z + 17 * 17 * 17 * 17 * 17) % 17);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_device_second_half() success" << std::endl;
}

void test_evaluate_poly_grid_device_p_squared() {
    Zn<float> ring{ 17 * 17 };
    const size_t d = 3;
    constexpr size_t m = 3;

    // 1 + x^2 + y^2 + 2xz + 3z^3
    std::vector<ZnEl> poly = { 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 };

    std::vector<ZnEl> out = evaluate_poly_parallel<float, m>(ring, { poly }, 0, 17 * 17 * 17, d, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[(x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17]) == (1 + x * x + y * y + 2 * x * z + 3 * z * z * z + 17 * 17 * 17 * 17) % (17 * 17));
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_device() success" << std::endl;
}

void test_evaluate_poly_grid_device_empty_poly() {
    Zn<float> ring{ 17 * 17 };
    const size_t d = 1;
    constexpr size_t m = 3;

    // 1 + x, 1, 0
    std::vector<std::vector<ZnEl>> polys = {
        { 1, 1, 0, 0 },
        { 1 },
        {}
    };

    std::vector<ZnEl> out = evaluate_poly_parallel<float, m>(ring, polys, 0, 17 * 17 * 17, d, 17);

    for (int z = -8; z < 9; ++z) {
        for (int y = -8; y < 9; ++y) {
            for (int x = -8; x < 9; ++x) {
                assert(ring.lift(out[((x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17) * 3]) == (1 + x + 17 * 17) % (17 * 17));
                assert(ring.lift(out[((x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17) * 3 + 1]) == 1);
                assert(ring.lift(out[((x + 8) + (y + 8) * 17 + (z + 8) * 17 * 17) * 3 + 2]) == 0);
            }
        }
    }

    std::cout << "test_evaluate_poly_grid_device_empty_poly() success" << std::endl;
}

typedef std::optional<size_t> SplitPolySelector;

template<typename LongFloat>
std::vector<ZnEl> read_input(const Zn<LongFloat>& ring, const char* hash, const size_t m, const size_t d, const size_t version_a, const size_t component, const SplitPolySelector k) {
    std::string filename = "E:\\polynomial_";
    filename += hash;
    filename += "_";
    filename += std::to_string(d);
    filename += "_";
    filename += std::to_string(m);
    filename += "_";
    filename += std::to_string(version_a);
    filename += "_";
    filename += std::to_string(component);
    if (k.has_value()) {
        filename += "_x";
        filename += std::to_string(k.value());
    }
    else {
        filename += "_const";
    }
    // "empty" polynomial 0 of degree -1
    if (d == component && k.has_value()) {
        return { };
    }
    const size_t degree = d - component - (k.has_value() ? 1 : 0);
    return read_polynomial(ring, filename.c_str(), degree, m);
}

std::string output_filename(const char* hash, const size_t m, const size_t d, const size_t version_a, const size_t p, const SplitPolySelector k) {
    std::string filename = "E:\\evaluations_";
    filename += hash;
    filename += "_";
    filename += std::to_string(d);
    filename += "_";
    filename += std::to_string(m);
    filename += "_";
    filename += std::to_string(version_a);
    filename += "_";
    filename += std::to_string(p);
    if (k.has_value()) {
        filename += "_x";
        filename += std::to_string(k.value());
    }
    else {
        filename += "_const";
    }
    return filename;
}

template<typename LongFloat>
void write_output(const Zn<LongFloat>& ring, const char* hash, const size_t m, const size_t d, const size_t version_a, const size_t p, const SplitPolySelector k, const std::vector<ZnEl>& data) {
    write_evaluations(ring, output_filename(hash, m, d, version_a, p, k).c_str(), data);
}

template<typename LongFloat, size_t m>
void run_for_modulus(const char* hash, const size_t d, const size_t poly_count, const uint32_t p, const uint32_t n, std::optional<size_t> k, const size_t component_count) {
    const Zn<LongFloat> ring{ n };

    for (size_t component_version = 0; component_version < component_count; ++component_version) {
        std::remove(output_filename(hash, m, d, component_version, n, k).c_str());

        std::vector<std::vector<ZnEl>> polynomials;
        for (size_t i = 0; i < poly_count; ++i) {
            polynomials.push_back(read_input<LongFloat>(ring, hash, m, d, component_version, i, k));
        }

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        uint64_t kernel_time = 0;
        const size_t max_evaluation_batch = ((1 << 30) / p) * p;
        for (size_t i = 0; i < static_pow<m>(p); i += max_evaluation_batch) {
            std::chrono::steady_clock::time_point kernel_start = std::chrono::steady_clock::now();
            const auto evaluations = evaluate_poly_parallel<LongFloat, m>(ring, polynomials, i, std::min(i + max_evaluation_batch, static_pow<m>(p)), d - (k.has_value() ? 1 : 0), p);
            std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
            kernel_time += std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count();

            write_output(ring, hash, m, d, component_version, p, k, evaluations);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Finished " << component_version << "/" << component_count << " for " << n << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        std::cout << "Computation took " << kernel_time << " ms" << std::endl;
    }
}

template<size_t m>
void run_suitable_for_modulus(const char* hash, const size_t d, const size_t poly_count, const uint32_t p, const uint32_t n, std::optional<size_t> k, const size_t component_count) {
    if (n < (1 << 11)) {
        run_for_modulus<float, m>(hash, d, poly_count, p, n, k, component_count);
    }
    else {
        run_for_modulus<double, m>(hash, d, poly_count, p, n, k, component_count);
    }
}

int main(int argc, char* argv[]) {

    try {

        constexpr size_t m = 3;

        if (argc == 1) {
            // later code
        }
        else if (argc == 8) {
            std::string hash(argv[1]);
            std::string degree(argv[2]);
            std::string poly_count(argv[3]);
            std::string prime(argv[4]);
            std::string split_variable(argv[5]);
            std::string component_count_str(argv[6]);
            std::string modulus(argv[7]);

            const size_t d = std::stoi(degree);
            const size_t c = std::stoi(poly_count);
            const size_t p = std::stoi(prime);
            const size_t n = std::stoi(modulus);
            const size_t component_count = std::stoi(component_count_str);

            // we make the assumption in many places that p is odd, in particular that we can iterate over -floor(p/2), ..., floor(p/2)
            // and get the p smallest elements
            if (n == 2) {
                throw std::invalid_argument("The prime p = 2 is not supported");
            }

            SplitPolySelector k;
            if (split_variable != "const") {
                k.emplace(std::stoi(split_variable));
            }
        
            run_suitable_for_modulus<m>(hash.c_str(), d, c, p, n, k, component_count);

            return 0;
        }
        else {
            throw std::invalid_argument("Either no arguments or three arguments (poly identifier, degree and max prime) are expected.");
        }

        //run_suitable_for_modulus<m>("ALmUQg", 10, 11, 13, std::nullopt);
    } catch (std::exception& exception) {
        std::cerr << "Aborting due to error: " << exception.what() << std::endl;
        return 1;
    }

    test_degrevlex_block_iter();
    test_evaluate_poly_grid();
    test_evaluate_poly_grid_order();
    test_evaluate_poly_grid_larger();
    test_evaluate_poly_grid_device();
    test_evaluate_poly_grid_larger_p_squared();
    test_evaluate_poly_grid_device_second_half();
    test_evaluate_poly_grid_device_empty_poly();
    test_evaluate_poly_grid_four_vars();
    return 0;
}
