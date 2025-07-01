#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "symbolic_evaluation.h"
#include "kernels.h"

__device__ int get_arity(int op)
{
    switch (op)
    {
    case SIN:
    case COS:
    case EXP:
    case LOG:
    case SQUARE:
    case SQRT:
        return 1;
    case ADD:
    case SUB:
    case MUL:
    case DIV:
        return 2;
    default:
        return 0;
    }
}

// Part 2: Forward Pass Kernel
// Kernel Tuning Parameters
constexpr size_t B_b = 16; // Tile dimension for batch
constexpr size_t N_b = 16; // Tile dimension for data points

__global__ void forward_step_k_kernel(
    float *cache, const int *ops, const int *ch, const float *x, const float *c, const int *ConstantPosition,
    size_t M, size_t B, size_t N, int n_x, int k)
{
    // --- a. Identify Indices ---
    size_t b_global = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t b_local = threadIdx.x;
    size_t n_local = threadIdx.y;

    if (b_global >= B || n_global >= N)
    {
        return;
    }

    // --- b. Cooperative Load ---
    __shared__ float tile_arg0[B_b][N_b];
    __shared__ float tile_arg1[B_b][N_b];

    size_t op_idx = (size_t)k * B + b_global;
    int op = ops[op_idx];

    // Load arguments into shared memory
    if (op != NO_OP)
    {
        int arity = get_arity(op);
        if (arity >= 1)
        {
            size_t child0_k = ch[op_idx * MAX_ARITY + 0];
            tile_arg0[b_local][n_local] = cache[child0_k * B * N + b_global * N + n_global];
        }
        if (arity == 2)
        {
            size_t child1_k = ch[op_idx * MAX_ARITY + 1];
            tile_arg1[b_local][n_local] = cache[child1_k * B * N + b_global * N + n_global];
        }
    }

    // --- c. Synchronize ---
    __syncthreads();

    // --- d. Compute ---
    float result = 0.0f;
    switch (op)
    {
    case NO_OP:
        break;
    case LEARNABLE_CONSTANT:
    {
        int c_idx = ConstantPosition[op_idx];
        if (c_idx != -1)
            result = c[c_idx];
        break;
    }
    case CONST_1:
        result = 1.0f;
        break;
    case SIN:
        result = sinf(tile_arg0[b_local][n_local]);
        break;
    case COS:
        result = cosf(tile_arg0[b_local][n_local]);
        break;
    case EXP:
        result = expf(tile_arg0[b_local][n_local]);
        break;
    case LOG:
        result = logf(fabsf(tile_arg0[b_local][n_local]) + 1e-8);
        break;
    case SQUARE:
    {
        float v = tile_arg0[b_local][n_local];
        result = v * v;
        break;
    }
    case SQRT:
        result = sqrtf(fabsf(tile_arg0[b_local][n_local]));
        break;
    case ADD:
        result = tile_arg0[b_local][n_local] + tile_arg1[b_local][n_local];
        break;
    case SUB:
        result = tile_arg0[b_local][n_local] - tile_arg1[b_local][n_local];
        break;
    case MUL:
        result = tile_arg0[b_local][n_local] * tile_arg1[b_local][n_local];
        break;
    case DIV:
        result = tile_arg0[b_local][n_local] / (tile_arg1[b_local][n_local] + 1e-8);
        break;
    default:
    { // Variable case
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            result = x[n_global * n_x + var_idx];
        }
    }
    }

    // --- e. Write Result ---
    if (op != NO_OP)
    {
        cache[(size_t)k * B * N + b_global * N + n_global] = result;
    }
}

// Part 3: Backward Pass Kernel
__global__ void backward_step_k_kernel(
    float *grad_cache, float *grad_c, float *grad_x,
    const float *cache, const int *ops, const int *ch, const int *ConstantPosition,
    size_t M, size_t B, size_t N, int n_x, int k)
{
    size_t b_global = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_global = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (b_global >= B || n_global >= N)
    {
        return;
    }

    size_t current_idx = (size_t)k * B * N + b_global * N + n_global;

    // --- a. Fetch Incoming Gradient ---
    float g_in = grad_cache[current_idx];
    if (g_in == 0.0f)
        return;

    // --- b. Identify Operation and Children ---
    size_t op_idx = (size_t)k * B + b_global;
    int op = ops[op_idx];

    // --- c. Apply Chain Rule and Accumulate Gradients ---
    switch (op)
    {
    case LEARNABLE_CONSTANT:
    {
        int c_idx = ConstantPosition[op_idx];
        if (c_idx != -1)
            atomicAdd(&grad_c[c_idx], g_in);
        break;
    }
    case SIN:
    {
        size_t child0_k = ch[op_idx * MAX_ARITY + 0];
        float child_val = cache[child0_k * B * N + b_global * N + n_global];
        float grad_out = g_in * cosf(child_val);
        atomicAdd(&grad_cache[child0_k * B * N + b_global * N + n_global], grad_out);
        break;
    }
    case COS:
    {
        size_t child0_k = ch[op_idx * MAX_ARITY + 0];
        float child_val = cache[child0_k * B * N + b_global * N + n_global];
        float grad_out = g_in * -sinf(child_val);
        atomicAdd(&grad_cache[child0_k * B * N + b_global * N + n_global], grad_out);
        break;
    }
    case MUL:
    {
        size_t child0_k = ch[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch[op_idx * MAX_ARITY + 1];
        float child0_val = cache[child0_k * B * N + b_global * N + n_global];
        float child1_val = cache[child1_k * B * N + b_global * N + n_global];
        atomicAdd(&grad_cache[child0_k * B * N + b_global * N + n_global], g_in * child1_val);
        atomicAdd(&grad_cache[child1_k * B * N + b_global * N + n_global], g_in * child0_val);
        break;
    }
    case ADD:
    {
        size_t child0_k = ch[op_idx * MAX_ARITY + 0];
        size_t child1_k = ch[op_idx * MAX_ARITY + 1];
        atomicAdd(&grad_cache[child0_k * B * N + b_global * N + n_global], g_in);
        atomicAdd(&grad_cache[child1_k * B * N + b_global * N + n_global], g_in);
        break;
    }
    // Add other operators here... (EXP, LOG, SQUARE, SQRT, SUB, DIV)
    // ...
    default:
    { // Variable case
        if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
        {
            size_t var_idx = op - VAR_START_ID;
            atomicAdd(&grad_x[n_global * n_x + var_idx], g_in);
        }
    }
    }
}

__global__ void validate_inputs_kernel(
    const int *ops, const int *ch, size_t M, size_t B, int *error_flag)
{
    size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || b >= B)
        return;

    // Early exit if an error has already been found by another thread.
    if (atomicCAS(error_flag, 0, 0) != 0)
        return;

    size_t op_idx = m * B + b;
    int op = ops[op_idx];

    if (op != NO_OP)
    {
        int expected_arity = get_arity(op);
        int actual_arity = 0;
        for (int i = 0; i < MAX_ARITY; ++i)
        {
            int child_k_int = ch[op_idx * MAX_ARITY + i];
            if (child_k_int != -1)
            {
                size_t child_k = child_k_int;
                actual_arity++;
                if (child_k >= m)
                {
                    atomicExch(error_flag, 1); // Error code 1 for invalid child index
                    return;
                }
            }
        }
        if (actual_arity != expected_arity)
        {
            atomicExch(error_flag, 2); // Error code 2 for arity mismatch
            return;
        }
    }
}

// --- Kernel Launchers ---

void launch_validate_inputs(const int *ops_ptr, const int *ch_ptr, size_t M, size_t B, int *error_flag_ptr)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((unsigned int)((M + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (unsigned int)((B + threadsPerBlock.y - 1) / threadsPerBlock.y));
    validate_inputs_kernel<<<numBlocks, threadsPerBlock>>>(ops_ptr, ch_ptr, M, B, error_flag_ptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in launch_validate_inputs: %s\n", cudaGetErrorString(err));
    }
}

void launch_forward_step_k(
    float *cache_ptr, const int64 *ops_ptr, const int64 *ch_ptr, const float *x_ptr,
    const float *Constants_ptr, const int64 *ConstantPosition_ptr,
    size_t M, size_t B, size_t N, size_t n_x, size_t k)
{
    dim3 threadsPerBlock(B_b, N_b);
    dim3 numBlocks((unsigned int)((B + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (unsigned int)((N + threadsPerBlock.y - 1) / threadsPerBlock.y));

    forward_step_k_kernel<<<numBlocks, threadsPerBlock>>>(
        cache_ptr, ops_ptr, ch_ptr, x_ptr, Constants_ptr, ConstantPosition_ptr,
        M, B, N, n_x, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in forward_step_k: %s\n", cudaGetErrorString(err));
    }
}

void launch_backward_step_k(
    float *grad_cache_ptr, float *grad_Constants_ptr, float *grad_x_ptr,
    const float *cache_ptr, const int *ops_ptr, const int *ch_ptr, const int *ConstantPosition_ptr,
    size_t M, size_t B, size_t N, int n_x, int k)
{
    dim3 threadsPerBlock(B_b, N_b);
    dim3 numBlocks((unsigned int)((B + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (unsigned int)((N + threadsPerBlock.y - 1) / threadsPerBlock.y));

    backward_step_k_kernel<<<numBlocks, threadsPerBlock>>>(
        grad_cache_ptr, grad_Constants_ptr, grad_x_ptr, cache_ptr, ops_ptr, ch_ptr, ConstantPosition_ptr,
        M, B, N, n_x, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in backward_step_k: %s\n", cudaGetErrorString(err));
    }
}