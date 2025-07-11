#include "evaluation_kernels.h"
#include "operators.h"
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/Atomic.cuh>

// --- CUDA Kernel Definitions ---

// Kernel Tuning Parameters
constexpr size_t B_b = 16; // Tile dimension for batch
constexpr size_t N_b = 16; // Tile dimension for data points

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

    int64_t op = ops_acc[m][b];

    int expected_arity = get_arity(op);
    int actual_arity = 0;
    for (int i = 0; i < MAX_ARITY; ++i)
    {
        int64_t child_k = ch_acc[m][b][i];
        if (child_k != NULL_CHILD)
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
__global__ void evaluation_forward_step_k_kernel(
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

    if (op == NO_OP)
    {
        cache_acc[k][n_global][b_global] = std::numeric_limits<scalar_t>::quiet_NaN();
        return;
    }

    scalar_t arg0 = static_cast<scalar_t>(0.0);
    scalar_t arg1 = static_cast<scalar_t>(0.0);

    const int arity = get_arity(op);
    if (arity >= 1)
    {
        int64_t ch0_idx = ch_acc[k][b_global][0];
        arg0 = cache_acc[ch0_idx][n_global][b_global];
    }
    if (arity == 2)
    {
        int64_t ch1_idx = ch_acc[k][b_global][1];
        arg1 = cache_acc[ch1_idx][n_global][b_global];
    }

    scalar_t result = static_cast<scalar_t>(0.0);
    // CORRECTED: Unified switch statement
    switch (op)
    {
    case NO_OP:
        result = std::numeric_limits<scalar_t>::quiet_NaN(); // Explicitly set result to NaN for NO_OP
        break;
    case LEARNABLE_CONSTANT:
    {
        int64_t c_idx = posC_acc[k][b_global];
        if (c_idx != NULL_CHILD)
            result = c_acc[c_idx];
        break;
    }
    case CONST_1:
        result = static_cast<scalar_t>(1.0);
        break;
    case CONST_2:
        result = static_cast<scalar_t>(2.0);
        break;
    case CONST_3:
        result = static_cast<scalar_t>(3.0);
        break;
    case CONST_4:
        result = static_cast<scalar_t>(4.0);
        break;
    case CONST_5:
        result = static_cast<scalar_t>(5.0);
        break;
    case PI:
        result = static_cast<scalar_t>(M_PI); // Use a more precise value for PI
        break;
    case E:
        result = static_cast<scalar_t>(M_E); // Use a more precise value for E
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
    case POW:
        result = pow_wrapper(arg0, arg1);
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
    cache_acc[k][n_global][b_global] = result;
}

template <typename scalar_t>
__global__ void evaluation_backward_step_k_kernel(
    torch::PackedTensorAccessor64<scalar_t, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> grad_c_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> grad_x_acc,
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k, int32_t *error_flag_ptr)
{
    size_t b_global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t B = ops_acc.size(1);
    const size_t N = grad_x_acc.size(0);

    if (b_global >= B || n_global >= N)
        return;

    scalar_t g_in = grad_cache_acc[k][n_global][b_global];

    int64_t op = ops_acc[k][b_global];

    // TODO
    // Check for NaN/inf in the incoming gradient. This is an error condition.
    //    if (isinf(g_in) || isnan(g_in))
    //    {
    //        atomicExch(error_flag_ptr, 1); // Use atomic exchange for error flag
    //        return;
    //    }

    if (op == NO_OP)
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        // If we have a non-zero gradient for a NO_OP, this is an error.
        // The forward pass should produce a zero, so the gradient should be zero unless there's a bug.
        atomicExch(error_flag_ptr, 2);
        return;
    }

    // If the operation is NO_OP, no gradient is needed, and g_in should be zero.

    switch (op)
    {
    case LEARNABLE_CONSTANT:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t c_idx = posC_acc[k][b_global];
        if (c_idx != NULL_CHILD)
        {
            gpuAtomicAdd(&grad_c_acc[c_idx], g_in);
        }
        break;
    }
    case SIN:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t g_out0 = g_in * cos_wrapper(arg0);
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    case COS:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t g_out0 = g_in * -sin_wrapper(arg0);
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    case EXP:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t out_val = cache_acc[k][n_global][b_global]; // Use cached value
        scalar_t g_out0 = g_in * out_val;
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    case LOG:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t g_out0;
        if (arg0 <= static_cast<scalar_t>(0.0))
        {
            atomicExch(error_flag_ptr, 3); // Error code 3 for gradient on invalid log
            g_out0 = static_cast<scalar_t>(0.0);
        }
        else
        {
            g_out0 = div_wrapper(g_in, arg0);
        }
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    case MUL:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        int64_t ch1_idx = ch_acc[k][b_global][1];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t arg1 = cache_acc[ch1_idx][n_global][b_global];
        scalar_t g_out0 = g_in * arg1;
        scalar_t g_out1 = g_in * arg0;
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        gpuAtomicAdd(&grad_cache_acc[ch1_idx][n_global][b_global], g_out1);
        break;
    }
    case ADD:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        int64_t ch1_idx = ch_acc[k][b_global][1];
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_in);
        gpuAtomicAdd(&grad_cache_acc[ch1_idx][n_global][b_global], g_in);
        break;
    }
    case SUB:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        int64_t ch1_idx = ch_acc[k][b_global][1];
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_in);
        gpuAtomicAdd(&grad_cache_acc[ch1_idx][n_global][b_global], -g_in);
        break;
    }
    case DIV:
    {
        // TODO: clean this up a bit
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        int64_t ch1_idx = ch_acc[k][b_global][1];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t arg1 = cache_acc[ch1_idx][n_global][b_global];
        scalar_t g_out0, g_out1;
        if (arg1 == static_cast<scalar_t>(0.0))
        {
            atomicExch(error_flag_ptr, 5); // Error code 5 for gradient on invalid div
            g_out0 = static_cast<scalar_t>(0.0);
            g_out1 = static_cast<scalar_t>(0.0);
        }
        else
        {
            g_out0 = div_wrapper(g_in, arg1);
            g_out1 = -div_wrapper(g_in * arg0, mul_wrapper(arg1, arg1));
        }
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        gpuAtomicAdd(&grad_cache_acc[ch1_idx][n_global][b_global], g_out1);
        break;
    }
    case POW:
    {
        // TODO: clean this up a bit
        int64_t ch0_idx = ch_acc[k][b_global][0];
        int64_t ch1_idx = ch_acc[k][b_global][1];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t arg1 = cache_acc[ch1_idx][n_global][b_global];
        scalar_t g_out0, g_out1;
        if (arg0 < static_cast<scalar_t>(0.0))
        {
            // atomicExch(error_flag_ptr, 6); // Error code 6 for gradient on invalid pow
            g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
            g_out1 = std::numeric_limits<scalar_t>::quiet_NaN();
        }
        else
        {
            if (g_in == static_cast<scalar_t>(0.0))
                return;
            g_out0 = mul_wrapper(g_in, mul_wrapper(pow_wrapper(arg0, sub_wrapper(arg1, static_cast<scalar_t>(1.0))), arg1));
            g_out1 = mul_wrapper(g_in, mul_wrapper(log_wrapper(arg0), pow_wrapper(arg0, arg1)));
        }
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        gpuAtomicAdd(&grad_cache_acc[ch1_idx][n_global][b_global], g_out1);
        break;
    }
    case SQUARE:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t g_out0 = g_in * 2 * arg0;
        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    case SQRT:
    {
        int64_t ch0_idx = ch_acc[k][b_global][0];
        scalar_t arg0 = cache_acc[ch0_idx][n_global][b_global];
        scalar_t g_out0;

        // TODO: think this through again
        // The commented-out code is consistent with the logic that if the g_in is zero, then the g_out0 should also be zero.
        // However, this is not consistent with pytorch's behavior, which returns NaN for gradients of sqrt at negative numbers even if g_in is zero.

        // if (g_in == static_cast<scalar_t>(0.0)) return;
        // if (arg0 <= static_cast<scalar_t>(0.0))
        //{
        //     atomicExch(error_flag_ptr, 4); // Error code 4 for gradient on invalid sqrt
        //     g_out0 = static_cast<scalar_t>(0.0);
        // }
        // else
        //{
        //     g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
        // }

        if (arg0 < static_cast<scalar_t>(0.0))
            g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
        else
        {
            if (g_in == static_cast<scalar_t>(0.0))
                g_out0 = static_cast<scalar_t>(0.0); // Return zero gradient if incoming gradient is zero
            else
                g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
        }

        gpuAtomicAdd(&grad_cache_acc[ch0_idx][n_global][b_global], g_out0);
        break;
    }
    default:
    {
        if (g_in == static_cast<scalar_t>(0.0))
            return;
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            gpuAtomicAdd(&grad_x_acc[n_global][var_idx], g_in);
        }
    }
    }
}

template <typename scalar_t>
void evaluation_forward_step_k_cuda_impl(
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

    evaluation_forward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        cache_acc, ops_acc, ch_acc, x_acc, c_acc, posC_acc, n_x, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void evaluation_backward_step_k_cuda_impl(
    torch::PackedTensorAccessor64<scalar_t, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> grad_c_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> grad_x_acc,
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = grad_x_acc.size(0);
    dim3 threadsPerBlock(B_b, N_b);
    dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    evaluation_backward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        grad_cache_acc, grad_c_acc, grad_x_acc, cache_acc, ops_acc, ch_acc, posC_acc, n_x, k, error_flag_ptr);
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
template __global__ void evaluation_forward_step_k_kernel<float>(
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<float, 2> x_acc,
    torch::PackedTensorAccessor64<float, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

template __global__ void evaluation_forward_step_k_kernel<double>(
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<double, 2> x_acc,
    torch::PackedTensorAccessor64<double, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k);

template __global__ void evaluation_backward_step_k_kernel<float>(
    torch::PackedTensorAccessor64<float, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<float, 1> grad_c_acc,
    torch::PackedTensorAccessor64<float, 2> grad_x_acc,
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k, int32_t *error_flag_ptr);

template __global__ void evaluation_backward_step_k_kernel<double>(
    torch::PackedTensorAccessor64<double, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<double, 1> grad_c_acc,
    torch::PackedTensorAccessor64<double, 2> grad_x_acc,
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    size_t n_x, size_t k, int32_t *error_flag_ptr);

// Explicit template instantiation for scalar types
template void evaluation_forward_step_k_cuda_impl<float>(
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<float, 2> x_acc,
    torch::PackedTensorAccessor64<float, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cuda_impl<float>(
    torch::PackedTensorAccessor64<float, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<float, 1> grad_c_acc,
    torch::PackedTensorAccessor64<float, 2> grad_x_acc,
    torch::PackedTensorAccessor64<float, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

template void evaluation_forward_step_k_cuda_impl<double>(
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<double, 2> x_acc,
    torch::PackedTensorAccessor64<double, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cuda_impl<double>(
    torch::PackedTensorAccessor64<double, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<double, 1> grad_c_acc,
    torch::PackedTensorAccessor64<double, 2> grad_x_acc,
    torch::PackedTensorAccessor64<double, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);
