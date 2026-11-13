// Official DeepSeek-V4.1-Flash-1017 numerical components; no model execution.
// Reference: inference/kernel.py SHA2561236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455.
// Copyright (c)2023 DeepSeek, MIT; see full reference attribution in src/dsv41/numerics.rs.
// Compile this isolated module with --ftz=false --fmad=false and a dedicated source ID.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef DSV41_NUMERICS_SOURCE_ID
#error "Define the separately hashed DSV41_NUMERICS_SOURCE_ID; do not reuse a legacy module stamp"
#endif

namespace {
constexpr unsigned ABI_VERSION = 1;
constexpr unsigned BAD_INPUT = 1;
constexpr unsigned BAD_SCALE = 2;
constexpr unsigned OVERFLOW = 4;
__device__ __forceinline__ void fail(unsigned* status, unsigned bit) { atomicOr(status, bit); }
__device__ __forceinline__ float ue8m0(unsigned char code) {
    return __uint_as_float(code == 0 ? (1u << 22) : ((unsigned)code << 23));
}
__device__ __forceinline__ float fp4(unsigned code) {
    const float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    // Exact official convert.py FP4_TABLE uses +0 for both zero codes.
    const float magnitude = values[code & 7u];
    return (code & 8u) && (code & 7u) ? -magnitude : magnitude;
}
__device__ __forceinline__ unsigned char fp8(float value) {
    return __nv_cvt_float_to_fp8(fminf(448.f, fmaxf(-448.f, value)),
        __NV_SATFINITE, __NV_E4M3);
}
template<int GROUP>
__device__ __forceinline__ void quant_body(
    unsigned char* out, unsigned char* scales, unsigned* status,
    const __nv_bfloat16* input, uint64_t groups) {
    const unsigned lane = threadIdx.x & 31u;
    const uint64_t group = (uint64_t)blockIdx.x * 4u + (threadIdx.x >> 5);
    if (group >= groups) return; // whole warp exits together
    float values[GROUP / 32];
    float maximum = 0.f;
    #pragma unroll
    for (int step = 0; step < GROUP / 32; ++step) {
        float x = __bfloat162float(input[group * GROUP + lane + step * 32]);
        if (!isfinite(x)) { fail(status, BAD_INPUT); x = 0.f; }
        values[step] = x;
        maximum = fmaxf(maximum, fabsf(x));
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    const float divided = __fmul_rn(fmaxf(maximum, 1e-4f), 1.f / 448.f);
    const unsigned bits = __float_as_uint(divided);
    const int exponent = (int)((bits >> 23) & 255u) - 127 + ((bits & 0x7fffffu) != 0);
    const unsigned code = (unsigned)(exponent + 127);
    if (!(divided > 0.f) || !isfinite(divided) || code == 0 || code >= 255) {
        fail(status, BAD_SCALE);
        if (lane == 0) scales[group] = 127;
        #pragma unroll
        for (int step = 0; step < GROUP / 32; ++step)
            out[group * GROUP + lane + step * 32] = 0;
        return;
    }
    const float scale = __uint_as_float(code << 23);
    if (lane == 0) scales[group] = (unsigned char)code;
    #pragma unroll
    for (int step = 0; step < GROUP / 32; ++step) {
        const float normalized = __fdiv_rn(values[step], scale);
        out[group * GROUP + lane + step * 32] = fp8(normalized);
    }
}
} // namespace

extern "C" __global__ void dsv41_numerics_identity(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = DSV41_NUMERICS_SOURCE_ID;
        out[1] = ABI_VERSION;
    }
}

extern "C" __global__ void dsv41_act_quant_group32(
    unsigned char* out, unsigned char* scales, unsigned* status,
    const __nv_bfloat16* input, uint64_t groups) {
    quant_body<32>(out, scales, status, input, groups);
}
// Diagnostic negative control only; no production switch or model route uses it.
extern "C" __global__ void dsv41_act_quant_group128_negative(
    unsigned char* out, unsigned char* scales, unsigned* status,
    const __nv_bfloat16* input, uint64_t groups) {
    quant_body<128>(out, scales, status, input, groups);
}

extern "C" __global__ void dsv41_fp8_encode_scalars(
    unsigned char* out, unsigned* status, const float* input, uint64_t count) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = input[index];
    if (!isfinite(value)) { fail(status, BAD_INPUT); value = 0.f; }
    out[index] = fp8(value);
}

extern "C" __global__ void dsv41_ue8m0_decode_scalars(
    float* out, unsigned* status, const unsigned char* input, uint64_t count) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned char code = input[index];
    if (code == 255) { fail(status, BAD_SCALE); out[index] = 0.f; return; }
    out[index] = ue8m0(code);
}

extern "C" __global__ void dsv41_fp4_native_dequant(
    float* out, unsigned* status, const unsigned char* packed,
    const unsigned char* scales, uint64_t count, uint64_t columns) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint64_t row = index / columns;
    const uint64_t column = index % columns;
    const unsigned char byte = packed[row * (columns / 2) + column / 2];
    const unsigned code = (column & 1u) ? byte >> 4 : byte & 15u;
    const unsigned char scale_code = scales[row * (columns / 32) + column / 32];
    if (scale_code == 255) { fail(status, BAD_SCALE); out[index] = 0.f; return; }
    const float value = __fmul_rn(fp4(code), ue8m0(scale_code));
    if (!isfinite(value)) { fail(status, OVERFLOW); out[index] = 0.f; return; }
    out[index] = value;
}

extern "C" __global__ void dsv41_rmsnorm_bf16(
    __nv_bfloat16* out, unsigned* status, const __nv_bfloat16* input,
    const float* weights, uint64_t rows, uint64_t columns, float epsilon) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    if (!(epsilon > 0.f) || !isfinite(epsilon)) { if (threadIdx.x == 0) fail(status, BAD_INPUT); return; }
    float sum = 0.f;
    for (uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = __bfloat162float(input[row * columns + column]);
        if (!isfinite(value)) { fail(status, BAD_INPUT); value = 0.f; }
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float mean = __fdiv_rn(partial[0], (float)columns);
    const float inverse = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, epsilon)));
    const bool good = isfinite(partial[0]) && isfinite(inverse);
    if (!good && threadIdx.x == 0) fail(status, OVERFLOW);
    for (uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = __bfloat162float(input[row * columns + column]);
        const float weight = weights[column];
        if (!isfinite(value) || !isfinite(weight)) { fail(status, BAD_INPUT); out[row * columns + column] = __float2bfloat16_rn(0.f); continue; }
        float result = good ? __fmul_rn(weight, __fmul_rn(value, inverse)) : 0.f;
        const __nv_bfloat16 rounded = __float2bfloat16_rn(result);
        if (!isfinite(result) || !isfinite(__bfloat162float(rounded))) {
            fail(status, OVERFLOW); out[row * columns + column] = __float2bfloat16_rn(0.f);
        } else out[row * columns + column] = rounded;
    }
}
