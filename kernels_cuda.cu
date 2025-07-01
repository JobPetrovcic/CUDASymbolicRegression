#include "kernels.h"
#include "symbolic_evaluation.h"
#include <ATen/native/cuda/Loops.cuh>

// --- CUDA Kernel Definitions ---

// Kernel Tuning Parameters
constexpr size_t B_b = 16; // Tile dimension for batch
constexpr size_t N_b = 16; // Tile dimension for data points

template <typename scalar_t>
__global__ void forward_step_k_kernel(
    scalar_t *cache_ptr, const int64_t *ops_ptr, const int64_t *ch_ptr,
    const scalar_t *x_ptr, const scalar_t *c_ptr, const int64_t *ConstantPosition_ptr,
    size_t M, size_t B, size_t N, size_t n_x, size_t k)
{
    size_t b_global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = blockIdx.y * blockDim.y + threadIdx.y;
    if (b_global >= B || n_global >= N)
        return;

    size_t b_local = threadIdx.x;
    size_t n_local = threadIdx.y;
    extern __shared__ scalar_t tile_args[];
    scalar_t *tile_arg0 = tile_args;
    scalar_t *tile_arg1 = &tile_args[B_b * N_b];
    size_t op_idx = k * B + b_global;
    int op = ops_ptr[op_idx];

    if (op != NO_OP)
    {
        int arity = get_arity(op);
        if (arity >= 1)
        {
            size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
            tile_arg0[b_local * N_b + n_local] = cache_ptr[child0_k * B * N + b_global * N + n_global];
        }
        if (arity == 2)
        {
            size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
            tile_arg1[b_local * N_b + n_local] = cache_ptr[child1_k * B * N + b_global * N + n_global];
        }
    }
    __syncthreads();

    scalar_t result = static_cast<scalar_t>(0.0);
    // CORRECTED: Unified switch statement
    switch (op)
    {
    case NO_OP:
        break; // Do nothing for NO_OP
    case LEARNABLE_CONSTANT:
    {
        int64_t c_idx = ConstantPosition_ptr[op_idx];
        if (c_idx != -1)
            result = c_ptr[c_idx];
        break;
    }
    case CONST_1:
        result = static_cast<scalar_t>(1.0);
        break;
    case SIN:
        result = sin_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case COS:
        result = cos_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case EXP:
        result = exp_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case LOG:
        result = log_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case SQUARE:
        result = square_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case SQRT:
        result = sqrt_wrapper(tile_arg0[b_local * N_b + n_local]);
        break;
    case ADD:
        // BUG FIX: Corrected arguments to wrapper
        result = add_wrapper(tile_arg0[b_local * N_b + n_local], tile_arg1[b_local * N_b + n_local]);
        break;
    case SUB:
        result = sub_wrapper(tile_arg0[b_local * N_b + n_local], tile_arg1[b_local * N_b + n_local]);
        break;
    case MUL:
        // BUG FIX: Corrected arguments to wrapper
        result = mul_wrapper(tile_arg0[b_local * N_b + n_local], tile_arg1[b_local * N_b + n_local]);
        break;
    case DIV:
        result = div_wrapper(tile_arg0[b_local * N_b + n_local], tile_arg1[b_local * N_b + n_local]);
        break;
    default:
    {
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            result = x_ptr[n_global * n_x + var_idx];
        }
    }
    }

    // BUG FIX: Only write to the cache if the operation was not a NO_OP.
    if (op != NO_OP)
    {
        cache_ptr[k * B * N + b_global * N + n_global] = result;
    }
}

template <typename scalar_t>
__global__ void backward_step_k_kernel(
    scalar_t *grad_cache_ptr, scalar_t *grad_c_ptr, scalar_t *grad_x_ptr,
    const scalar_t *cache_ptr, const int64_t *ops_ptr, const int64_t *ch_ptr,
    const int64_t *ConstantPosition_ptr, size_t M, size_t B, size_t N, size_t n_x, size_t k)
{
    size_t b_global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = blockIdx.y * blockDim.y + threadIdx.y;
    if (b_global >= B || n_global >= N)
        return;

    size_t current_idx = k * B * N + b_global * N + n_global;
    scalar_t g_in = grad_cache_ptr[current_idx];
    if (g_in == static_cast<scalar_t>(0.0))
        return;

    size_t op_idx = k * B + b_global;
    int op = ops_ptr[op_idx];

    switch (op)
    {
    case LEARNABLE_CONSTANT:
    {
        int64_t c_idx = ConstantPosition_ptr[op_idx];
        if (c_idx != -1)
            at::atomicAdd(&grad_c_ptr[c_idx], g_in);
        break;
    }
    case SIN:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        scalar_t child_val = cache_ptr[child0_k * B * N + b_global * N + n_global];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in * cos_wrapper(child_val));
        break;
    }
    case COS:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        scalar_t child_val = cache_ptr[child0_k * B * N + b_global * N + n_global];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in * -sin_wrapper(child_val));
        break;
    }
    case MUL:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
        scalar_t child0_val = cache_ptr[child0_k * B * N + b_global * N + n_global];
        scalar_t child1_val = cache_ptr[child1_k * B * N + b_global * N + n_global];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in * child1_val);
        at::atomicAdd(&grad_cache_ptr[child1_k * B * N + b_global * N + n_global], g_in * child0_val);
        break;
    }
    case ADD:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in);
        at::atomicAdd(&grad_cache_ptr[child1_k * B * N + b_global * N + n_global], g_in);
        break;
    }
    case SUB:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in);
        at::atomicAdd(&grad_cache_ptr[child1_k * B * N + b_global * N + n_global], -g_in);
        break;
    }
    case DIV:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
        scalar_t child0_val = cache_ptr[child0_k * B * N + b_global * N + n_global];
        scalar_t child1_val = cache_ptr[child1_k * B * N + b_global * N + n_global];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], div_wrapper(g_in, child1_val));
        at::atomicAdd(&grad_cache_ptr[child1_k * B * N + b_global * N + n_global], -mul_wrapper(g_in, div_wrapper(child0_val, square_wrapper(child1_val))));
        break;
    }
    case SQUARE:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        scalar_t child0_val = cache_ptr[child0_k * B * N + b_global * N + n_global];
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], g_in * static_cast<scalar_t>(2.0) * child0_val);
        break;
    }
    case SQRT:
    {
        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
        scalar_t out_val = cache_ptr[current_idx];
        scalar_t grad_out = div_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), out_val));
        at::atomicAdd(&grad_cache_ptr[child0_k * B * N + b_global * N + n_global], grad_out);
        break;
    }
    default:
    {
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            at::atomicAdd(&grad_x_ptr[n_global * n_x + var_idx], g_in);
        }
    }
    }
}

__global__ void validate_inputs_kernel(
    const int64_t *ops_ptr, const int64_t *ch_ptr, size_t M, size_t B, int32_t *error_flag_ptr)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || b >= B)
        return;
    if (atomicCAS(error_flag_ptr, 0, 0) != 0)
        return;

    size_t op_idx = m * B + b;
    int op = ops_ptr[op_idx];

    int expected_arity = get_arity(op);
    int actual_arity = 0;
    for (int i = 0; i < MAX_ARITY; ++i)
    {
        int64_t child_k = ch_ptr[op_idx * MAX_ARITY + i];
        if (child_k != -1)
        {
            actual_arity++;
            if (child_k >= m)
            {
                atomicExch(error_flag_ptr, 1); // Error: Invalid child index
                return;
            }
        }
    }
    if (actual_arity != expected_arity)
    {
        atomicExch(error_flag_ptr, 2); // Error: Arity mismatch
        return;
    }
}

void forward_step_k_impl(
    torch::Tensor cache, const torch::Tensor &ops, const torch::Tensor &ch,
    const torch::Tensor &x, const torch::Tensor &Constants,
    const torch::Tensor &ConstantPosition, int64_t k)
{

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(), "forward_step_k_cuda", ([&]
                                                                                     {
        const auto M = ops.size(0), B = ops.size(1), N = x.size(0), n_x = x.size(1);
        dim3 threadsPerBlock(B_b, N_b);
        dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        size_t smem_size = 2 * B_b * N_b * sizeof(scalar_t); // For two tiles

        forward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock, smem_size>>>(
            cache.data_ptr<scalar_t>(), ops.data_ptr<int64_t>(), ch.data_ptr<int64_t>(),
            x.data_ptr<scalar_t>(), Constants.data_ptr<scalar_t>(), ConstantPosition.data_ptr<int64_t>(),
            M, B, N, n_x, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
}

void backward_step_k_impl(
    torch::Tensor grad_cache, torch::Tensor grad_Constants, torch::Tensor grad_x,
    const torch::Tensor &cache, const torch::Tensor &ops, const torch::Tensor &ch,
    const torch::Tensor &ConstantPosition, int64_t k)
{

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_cache.scalar_type(), "backward_step_k_cuda", ([&]
                                                                                           {
        const auto M = ops.size(0), B = ops.size(1), N = cache.size(2), n_x = grad_x.size(1);
        dim3 threadsPerBlock(B_b, N_b);
        dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        backward_step_k_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            grad_cache.data_ptr<scalar_t>(), grad_Constants.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
            cache.data_ptr<scalar_t>(), ops.data_ptr<int64_t>(), ch.data_ptr<int64_t>(), ConstantPosition.data_ptr<int64_t>(),
            M, B, N, n_x, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK(); }));
}

void validate_inputs_impl(const torch::Tensor &ops, const torch::Tensor &ch, torch::Tensor &error_flag)
{
    const auto M = ops.size(0), B = ops.size(1);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (B + threadsPerBlock.y - 1) / threadsPerBlock.y);

    validate_inputs_kernel<<<numBlocks, threadsPerBlock>>>(
        ops.data_ptr<int64_t>(), ch.data_ptr<int64_t>(), M, B, error_flag.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}