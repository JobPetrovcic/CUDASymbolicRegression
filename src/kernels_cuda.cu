#include "kernels.h"
#include "operators.h"
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/Atomic.cuh>

// TODO: faster memory loading in kernels

// --- CUDA Kernel Definitions ---

// Kernel Tuning Parameters
constexpr size_t B_b = 16; // Tile dimension for batch
constexpr size_t N_b = 16; // Tile dimension for data points

template <typename scalar_t>
__global__ void forward_step_k_kernel(
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> x_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k)
{
    size_t b_global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t B = ops_acc.size(1);
    const size_t N = x_acc.size(0);

    if (b_global >= B || n_global >= N)
        return;

    int64_t op = ops_acc[k][b_global];
    scalar_t arg0 = static_cast<scalar_t>(0.0);
    scalar_t arg1 = static_cast<scalar_t>(0.0);

    if (op != NO_OP)
    {
        int arity = get_arity(op);
        if (arity >= 1)
        {
            int64_t child0_k = ch_acc[k][b_global][0];
            arg0 = cache_acc[child0_k][b_global][n_global];
        }
        if (arity == 2)
        {
            int64_t child1_k = ch_acc[k][b_global][1];
            arg1 = cache_acc[child1_k][b_global][n_global];
        }
    }

    scalar_t result = static_cast<scalar_t>(0.0);
    // CORRECTED: Unified switch statement
    switch (op)
    {
    case NO_OP:
        break; // Do nothing for NO_OP
    case LEARNABLE_CONSTANT:
    {
        int64_t c_idx = posC_acc[k][b_global];
        if (c_idx != -1)
            result = c_acc[c_idx];
        break;
    }
    case CONST_1:
        result = static_cast<scalar_t>(1.0);
        break;
    case SIN:
        result = sin_wrapper(arg0);
        break;
    case COS:
        result = cos_wrapper(arg0);
        break;
    case EXP:
        result = exp_wrapper(arg0);
        break;
    case LOG:
        result = log_wrapper(arg0);
        break;
    case SQUARE:
        result = square_wrapper(arg0);
        break;
    case SQRT:
        result = sqrt_wrapper(arg0);
        break;
    case ADD:
        result = add_wrapper(arg0, arg1);
        break;
    case SUB:
        result = sub_wrapper(arg0, arg1);
        break;
    case MUL:
        result = mul_wrapper(arg0, arg1);
        break;
    case DIV:
        result = div_wrapper(arg0, arg1);
        break;
    default:
    {
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            result = x_acc[n_global][var_idx];
        }
    }
    }

    // BUG FIX: Only write to the cache if the operation was not a NO_OP.
    if (op != NO_OP)
    {
        cache_acc[k][b_global][n_global] = result;
    }
}

template <typename scalar_t>
__global__ void backward_step_k_kernel(
    torch::PackedTensorAccessor64<scalar_t, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> grad_c_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> grad_x_acc,
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k)
{
    size_t b_global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t B = ops_acc.size(1);
    const size_t N = grad_x_acc.size(0);

    if (b_global >= B || n_global >= N)
        return;

    scalar_t g_in = grad_cache_acc[k][b_global][n_global];
    if (g_in == static_cast<scalar_t>(0.0))
        return;

    int64_t op = ops_acc[k][b_global];

    switch (op)
    {
    case LEARNABLE_CONSTANT:
    {
        int64_t c_idx = posC_acc[k][b_global];
        if (c_idx != -1)
        {
            gpuAtomicAdd(&grad_c_acc[c_idx], g_in);
        }
        break;
    }
    case SIN:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        scalar_t child_val = cache_acc[child0_k][b_global][n_global];
        scalar_t grad_out = mul_wrapper(g_in, cos_wrapper(child_val));
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], grad_out);
        break;
    }
    case COS:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        scalar_t child_val = cache_acc[child0_k][b_global][n_global];
        scalar_t grad_out = mul_wrapper(g_in, -sin_wrapper(child_val));
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], grad_out);
        break;
    }
    case MUL:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        int64_t child1_k = ch_acc[k][b_global][1];
        scalar_t child0_val = cache_acc[child0_k][b_global][n_global];
        scalar_t child1_val = cache_acc[child1_k][b_global][n_global];
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], mul_wrapper(g_in, child1_val));
        gpuAtomicAdd(&grad_cache_acc[child1_k][b_global][n_global], mul_wrapper(g_in, child0_val));
        break;
    }
    case ADD:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        int64_t child1_k = ch_acc[k][b_global][1];
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], g_in);
        gpuAtomicAdd(&grad_cache_acc[child1_k][b_global][n_global], g_in);
        break;
    }
    case SUB:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        int64_t child1_k = ch_acc[k][b_global][1];
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], g_in);
        gpuAtomicAdd(&grad_cache_acc[child1_k][b_global][n_global], -g_in);
        break;
    }
    case DIV:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        int64_t child1_k = ch_acc[k][b_global][1];
        scalar_t child0_val = cache_acc[child0_k][b_global][n_global];
        scalar_t child1_val = cache_acc[child1_k][b_global][n_global];
        scalar_t grad_out0 = div_wrapper(g_in, child1_val);
        scalar_t grad_out1 = mul_wrapper(g_in, div_wrapper(-child0_val, mul_wrapper(child1_val, child1_val)));
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], grad_out0);
        gpuAtomicAdd(&grad_cache_acc[child1_k][b_global][n_global], grad_out1);
        break;
    }
    case SQUARE:
    {
        int64_t child0_k = ch_acc[k][b_global][0];
        scalar_t child_val = cache_acc[child0_k][b_global][n_global];
        scalar_t grad_out = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), child_val));
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], grad_out);
        break;
    }
    case SQRT:
    {
        scalar_t out_val = cache_acc[k][b_global][n_global];
        scalar_t grad_out = div_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), out_val));
        int64_t child0_k = ch_acc[k][b_global][0];
        gpuAtomicAdd(&grad_cache_acc[child0_k][b_global][n_global], grad_out);
        break;
    }
    default:
    {
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            gpuAtomicAdd(&grad_x_acc[n_global][var_idx], g_in);
        }
    }
    }
}

__global__ void validate_inputs_kernel(
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    int32_t *error_flag_ptr)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t M = ops_acc.size(0);
    const size_t B = ops_acc.size(1);

    if (m >= M || b >= B)
        return;

    // Ensure only one thread can set the error flag
    if (atomicCAS(error_flag_ptr, 0, 0) != 0)
        return;

    int64_t op = ops_acc[m][b];

    int expected_arity = get_arity(op);
    int actual_arity = 0;
    for (int i = 0; i < MAX_ARITY; ++i)
    {
        int64_t child_k = ch_acc[m][b][i];
        if (child_k != -1)
        {
            actual_arity++;
            if (child_k >= m)
            {
                atomicCAS(error_flag_ptr, 0, 1);
                return;
            }
        }
    }

    if (actual_arity != expected_arity)
    {
        atomicCAS(error_flag_ptr, 0, 1);
    }
}

template <typename scalar_t>
void forward_step_k_cuda_impl(
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> x_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = x_acc.size(0);
    dim3 threadsPerBlock(B_b, N_b);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    forward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        cache_acc, ops_acc, ch_acc, x_acc, c_acc, posC_acc, n_x, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void backward_step_k_cuda_impl(
    torch::PackedTensorAccessor64<scalar_t, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> grad_c_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> grad_x_acc,
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = grad_x_acc.size(0);
    dim3 threadsPerBlock(B_b, N_b);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    backward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        grad_cache_acc, grad_c_acc, grad_x_acc, cache_acc, ops_acc, ch_acc, posC_acc, n_x, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_inputs_cuda_impl(
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    int32_t *error_flag_ptr)
{
    const int64_t M = ops_acc.size(0);
    const int64_t B = ops_acc.size(1);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B + threadsPerBlock.y - 1) / threadsPerBlock.y);

    validate_inputs_kernel<<<numBlocks, threadsPerBlock>>>(
        ops_acc, ch_acc, error_flag_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicitly instantiate templates for float and double
template __global__ void forward_step_k_kernel<float>(
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<float, 2> x_acc,
    torch::PackedTensorAccessor64<float, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

template __global__ void forward_step_k_kernel<double>(
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<double, 2> x_acc,
    torch::PackedTensorAccessor64<double, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

template __global__ void backward_step_k_kernel<float>(
    torch::PackedTensorAccessor64<float, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<float, 1> grad_c_acc,
    torch::PackedTensorAccessor64<float, 2> grad_x_acc,
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

template __global__ void backward_step_k_kernel<double>(
    torch::PackedTensorAccessor64<double, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<double, 1> grad_c_acc,
    torch::PackedTensorAccessor64<double, 2> grad_x_acc,
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

// Explicit template instantiation for scalar types
template void forward_step_k_cuda_impl<float>(
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<float, 2> x_acc,
    torch::PackedTensorAccessor64<float, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void backward_step_k_cuda_impl<float>(
    torch::PackedTensorAccessor64<float, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<float, 1> grad_c_acc,
    torch::PackedTensorAccessor64<float, 2> grad_x_acc,
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void forward_step_k_cuda_impl<double>(
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<double, 2> x_acc,
    torch::PackedTensorAccessor64<double, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void backward_step_k_cuda_impl<double>(
    torch::PackedTensorAccessor64<double, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<double, 1> grad_c_acc,
    torch::PackedTensorAccessor64<double, 2> grad_x_acc,
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);
