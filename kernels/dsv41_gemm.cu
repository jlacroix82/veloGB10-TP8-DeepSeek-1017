// Official DeepSeek-V4.1-Flash-1017 native group32 GEMM components, no serving route.
// Reference kernel.py SHA2561236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455.
// DeepSeek reference copyright (c)2023 DeepSeek, MIT; attribution in dsv41/numerics.rs.
// Native e4m3 MMA fragment mapping follows gpu_batch.cu's independently qualified
// F8 parity lane; scales, ordering and layouts here are1017-specific.
// Compile separately for sm_121 with --ftz=false --fmad=false and a dedicated stamp.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_GEMM_SOURCE_ID
#error "A dedicated DSV41_GEMM_SOURCE_ID is required"
#endif
namespace {
constexpr unsigned ABI_VERSION = 1;
constexpr unsigned BAD_INPUT = 1, BAD_SCALE = 2, OVERFLOW = 4;
__device__ __forceinline__ void bad(unsigned* status, unsigned bit) { atomicOr(status, bit); }
__device__ __forceinline__ float scale(unsigned char code, unsigned* status) {
    if (code == 255) { bad(status, BAD_SCALE); return 1.f; }
    return __uint_as_float(code == 0 ? (1u << 22) : ((unsigned)code << 23));
}
__device__ __forceinline__ unsigned char finite_fp8(unsigned char code, unsigned* status) {
    if ((code & 127u) == 127u) { bad(status, BAD_INPUT); return 0; }
    return code;
}
__device__ __forceinline__ unsigned finite_fp8_word(const unsigned char* input, unsigned* status) {
    // Native matrices have K divisible by32; every caller's column is divisible
    // by4 and CUDA allocations are aligned. Preserve every finite code,
    // including signed zero, and the scalar NaN rejection in each byte lane.
    const unsigned word = *reinterpret_cast<const unsigned*>(input);
    const unsigned invalid = __vcmpeq4(word & 0x7f7f7f7fu, 0x7f7f7f7fu);
    if (invalid) bad(status, BAD_INPUT);
    return word & ~invalid;
}
__device__ __forceinline__ unsigned char fp4_as_fp8(unsigned nibble) {
    // Exact E2M1 values {0,.5,1,1.5,2,3,4,6}, all exactly representable in E4M3.
    // Official convert.py maps both FP4 zero codes to +0.
    unsigned magnitude = nibble & 7u;
    unsigned code = magnitude == 0 ? 0 : (magnitude == 1 ? 0x30 : (magnitude == 2 ? 0x38 :
        (magnitude == 3 ? 0x3c : (magnitude == 4 ? 0x40 : (magnitude == 5 ? 0x44 :
        (magnitude == 6 ? 0x48 : 0x4c))))));
    return (unsigned char)(code | ((magnitude && (nibble & 8u)) ? 0x80 : 0));
}
// Lossless register decode of four original packed E2M1 nibbles. Two bytewise
// equality masks handle the zero and .5 exceptions; all larger magnitudes are
// exactly 0x30 + 4*m in E4M3. Both FP4 zero codes remain canonical +0.
__device__ __forceinline__ unsigned fp4_word_as_fp8(unsigned raw) {
    unsigned spread = (raw | (raw << 8)) & 0x00ff00ffu;
    spread = (spread | (spread << 4)) & 0x0f0f0f0fu;
    unsigned magnitude = spread & 0x07070707u;
    unsigned zero = __vcmpeq4(magnitude, 0u);
    unsigned half = __vcmpeq4(magnitude, 0x01010101u);
    unsigned codes = (magnitude << 2) + 0x30303030u;
    codes -= half & 0x04040404u;
    return (codes | ((spread & 0x08080808u) << 4)) & ~zero;
}
__device__ __forceinline__ void mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ unsigned activation_word(
    const unsigned char* input, int row, int rows, int inner, int column, unsigned* status) {
    if (row >= rows) return 0; // Padded tokens never read outside the supplied matrix.
    return finite_fp8_word(input + (uint64_t)row * inner + column, status);
}
template<bool FP4>
__device__ __forceinline__ unsigned weight_word(
    const unsigned char* weights, int row, int outputs, int inner, int column, unsigned* status) {
    if (row >= outputs) return 0;
    if (FP4) {
        // inner%32==0 and column%4==0: every two-byte read is aligned and
        // wholly within this row and K32 group; no persistent repack or cast.
        const unsigned short* packed = reinterpret_cast<const unsigned short*>(
            weights + (uint64_t)row*(inner/2) + column/2);
        return fp4_word_as_fp8(*packed);
    }
    return finite_fp8_word(weights + (uint64_t)row * inner + column, status);
}
// Every warp owns its output tile's ENTIRE K chain in ascending group32 order.
// Token tiles are always16 wide; width changes only mask loads/stores. There is
// no split-K, cross-warp reduction, atomics on output, or batch-dependent GEMM choice.
template<bool FP4>
__device__ __forceinline__ void gemm_body(
    float* out_f32, __nv_bfloat16* out_bf16, unsigned* status,
    const unsigned char* activations, const unsigned char* activation_scales,
    const unsigned char* weights, const unsigned char* weight_scales,
    int rows, int outputs, int inner) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int output0 = ((int)blockIdx.x * 8 + warp) * 16;
    if (output0 >= outputs) return; // whole warp exits together
    const int token0 = (int)blockIdx.y * 16;
    const int groups = inner / 32;
    float accum[2][4] = {{0,0,0,0},{0,0,0,0}};
    for (int k = 0; k < groups; ++k) {
        const int column = k*32 + 4*t;
        unsigned a[4] = {
            weight_word<FP4>(weights, output0+g, outputs, inner, column, status),
            weight_word<FP4>(weights, output0+g+8, outputs, inner, column, status),
            weight_word<FP4>(weights, output0+g, outputs, inner, column+16, status),
            weight_word<FP4>(weights, output0+g+8, outputs, inner, column+16, status)
        };
        float raw[2][4] = {{0,0,0,0},{0,0,0,0}};
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            unsigned b[2] = {
                activation_word(activations, token0+sub*8+g, rows, inner, column, status),
                activation_word(activations, token0+sub*8+g, rows, inner, column+16, status)
            };
            mma(raw[sub], a, b);
        }
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int output = output0+g+8*(i>=2);
                const int token = token0+sub*8+2*t+(i&1);
                if (output < outputs && token < rows) {
                    float as = scale(activation_scales[(uint64_t)token*groups+k], status);
                    const int scale_row = FP4 ? output : output/32;
                    float ws = scale(weight_scales[(uint64_t)scale_row*groups+k], status);
                    // Preserve the official source's raw * activation_scale * weight_scale
                    // order. Keep the new K32 raw accumulator distinct and reset each iteration.
                    float corrected = __fmul_rn(__fmul_rn(raw[sub][i], as), ws);
                    float next = __fadd_rn(accum[sub][i], corrected);
                    if (!isfinite(corrected) || !isfinite(next)) {
                        bad(status, OVERFLOW); next = 0.f;
                    }
                    accum[sub][i] = next;
                }
            }
    }
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int output = output0+g+8*(i>=2);
            const int token = token0+sub*8+2*t+(i&1);
            if (output < outputs && token < rows) {
                const uint64_t index = (uint64_t)token*outputs+output;
                if (out_f32) out_f32[index] = accum[sub][i];
                if (out_bf16) {
                    __nv_bfloat16 value = __float2bfloat16_rn(accum[sub][i]);
                    if (!isfinite(__bfloat162float(value))) {
                        bad(status, OVERFLOW); value = __float2bfloat16_rn(0.f);
                    }
                    out_bf16[index] = value;
                }
            }
        }
}
} // namespace
extern "C" __global__ void dsv41_gemm_identity(unsigned long long* out) {
    if (!blockIdx.x && !threadIdx.x) { out[0]=DSV41_GEMM_SOURCE_ID; out[1]=ABI_VERSION; }
}
extern "C" __global__ void dsv41_gemm_fp8_group32(
    float* out_f32, __nv_bfloat16* out_bf16, unsigned* status,
    const unsigned char* activations, const unsigned char* activation_scales,
    const unsigned char* weights, const unsigned char* weight_scales,
    int rows, int outputs, int inner) {
    gemm_body<false>(out_f32,out_bf16,status,activations,activation_scales,weights,weight_scales,
        rows,outputs,inner);
}
extern "C" __global__ void dsv41_gemm_fp4_group32(
    float* out_f32, __nv_bfloat16* out_bf16, unsigned* status,
    const unsigned char* activations, const unsigned char* activation_scales,
    const unsigned char* weights, const unsigned char* weight_scales,
    int rows, int outputs, int inner) {
    gemm_body<true>(out_f32,out_bf16,status,activations,activation_scales,weights,weight_scales,
        rows,outputs,inner);
}

// Separate opt-in backend. Parallelism is across ORIGINAL K32 groups; no
// cross-group sum occurs here. Every contribution retains the original MMA,
// raw*activation_scale*weight_scale rounding, NaN rejection and scale checks.
// The caller owns groups*outputs*rows F32 entries until the final fence.
// Column-major within each group coalesces MMA fragment stores and ordered
// finish reads; only the final, once-per-output write uses model row-major.
extern "C" __global__ void dsv41_gemm_fp8_group32_contributions(
    float* contributions, unsigned* status,
    const unsigned char* activations, const unsigned char* activation_scales,
    const unsigned char* weights, const unsigned char* weight_scales,
    int rows, int outputs, int inner) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int output0 = ((int)blockIdx.x * 8 + warp) * 16;
    if (output0 >= outputs) return;
    const int token0 = (int)blockIdx.y * 16;
    const int groups = inner / 32;
    const int k = (int)blockIdx.z;
    if (k >= groups) return;
    const int column = k * 32 + 4 * t;
    unsigned a[4] = {
        weight_word<false>(weights, output0+g, outputs, inner, column, status),
        weight_word<false>(weights, output0+g+8, outputs, inner, column, status),
        weight_word<false>(weights, output0+g, outputs, inner, column+16, status),
        weight_word<false>(weights, output0+g+8, outputs, inner, column+16, status)
    };
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        unsigned b[2] = {
            activation_word(activations, token0+sub*8+g, rows, inner, column, status),
            activation_word(activations, token0+sub*8+g, rows, inner, column+16, status)
        };
        float raw[4] = {0,0,0,0};
        mma(raw, a, b);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int output = output0+g+8*(i>=2);
            const int token = token0+sub*8+2*t+(i&1);
            if (output < outputs && token < rows) {
                const float as = scale(activation_scales[(uint64_t)token*groups+k], status);
                const float ws = scale(weight_scales[(uint64_t)(output/32)*groups+k], status);
                // Keep nonfinite intermediates literally: the ORIGINAL
                // per-add overflow/reset rule executes in the ordered kernel.
                const float corrected = __fmul_rn(__fmul_rn(raw[i], as), ws);
                contributions[((uint64_t)k*outputs+output)*rows+token] = corrected;
            }
        }
    }
}

// One thread owns each complete output chain in exactly the original group
// order. In particular, an overflow resets THIS running sum before the next
// group; replacing this with a tree or a partial-sum reduction is forbidden.
extern "C" __global__ void dsv41_gemm_group32_ordered_finish(
    float* out_f32, __nv_bfloat16* out_bf16, unsigned* status,
    const float* contributions, int rows, int outputs, int inner) {
    const uint64_t index = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t count = (uint64_t)rows*outputs;
    if (index >= count) return;
    float accum = 0.f;
    for (int k = 0; k < inner/32; ++k) {
        const float corrected = contributions[(uint64_t)k*count+index];
        float next = __fadd_rn(accum, corrected);
        if (!isfinite(corrected) || !isfinite(next)) {
            bad(status, OVERFLOW); next = 0.f;
        }
        accum = next;
    }
    const uint64_t output_index = (index % rows)*outputs + index/rows;
    if (out_f32) out_f32[output_index] = accum;
    if (out_bf16) {
        __nv_bfloat16 value = __float2bfloat16_rn(accum);
        if (!isfinite(__bfloat162float(value))) {
            bad(status, OVERFLOW); value = __float2bfloat16_rn(0.f);
        }
        out_bf16[output_index] = value;
    }
}
