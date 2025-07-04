#include "kernels.h"
#include "operators.h"
#include <omp.h>  // For CPU parallelization
#include <atomic> // For std::atomic_exchange

// --- Forward Pass Implementation (CPU) ---

template <typename scalar_t>
void forward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<scalar_t, 2> x_acc,
    torch::TensorAccessor<scalar_t, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = x_acc.size(0);

// Parallelize the loops over the batch (B) and data (N) dimensions
#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            const int op = ops_acc[k][b];

            // Skip NO_OP entirely, leaving the cache value unchanged.
            if (op == NO_OP)
            {
                continue;
            }

            scalar_t arg0 = static_cast<scalar_t>(0.0);
            scalar_t arg1 = static_cast<scalar_t>(0.0);

            const int arity = get_arity(op);
            if (arity >= 1)
            {
                int64_t child0_k = ch_acc[k][b][0];
                arg0 = cache_acc[child0_k][b][n];
            }
            if (arity == 2)
            {
                int64_t child1_k = ch_acc[k][b][1];
                arg1 = cache_acc[child1_k][b][n];
            }

            scalar_t result = static_cast<scalar_t>(0.0);
            switch (op)
            {
            case LEARNABLE_CONSTANT:
            {
                int64_t c_idx = posC_acc[k][b];
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
                    result = x_acc[n][var_idx];
                }
            }
            }
            cache_acc[k][b][n] = result;
        }
    }
}

// --- Backward Pass Implementation (CPU) ---

template <typename scalar_t>
void backward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 3> grad_cache_acc,
    torch::TensorAccessor<scalar_t, 1> grad_c_acc,
    torch::TensorAccessor<scalar_t, 2> grad_x_acc,
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = grad_x_acc.size(0);

#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            const scalar_t g_in = grad_cache_acc[k][b][n];

            // Optimization: if incoming gradient is zero, no work to do.
            if (g_in == static_cast<scalar_t>(0.0))
            {
                continue;
            }

            const int op = ops_acc[k][b];

            switch (op)
            {
            case LEARNABLE_CONSTANT:
            {
                int64_t c_idx = posC_acc[k][b];
                if (c_idx != -1)
                {
#pragma omp atomic
                    grad_c_acc[c_idx] += g_in;
                }
                break;
            }
            case SIN:
            {
                int64_t child0_k = ch_acc[k][b][0];
                scalar_t child_val = cache_acc[child0_k][b][n];
                scalar_t grad_out = mul_wrapper(g_in, cos_wrapper(child_val));
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += grad_out;
                break;
            }
            case COS:
            {
                int64_t child0_k = ch_acc[k][b][0];
                scalar_t child_val = cache_acc[child0_k][b][n];
                scalar_t grad_out = mul_wrapper(g_in, -cos_wrapper(child_val));
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += grad_out;
                break;
            }
            case SQUARE:
            {
                int64_t child0_k = ch_acc[k][b][0];
                scalar_t child_val = cache_acc[child0_k][b][n];
                scalar_t grad_out = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), child_val));
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += grad_out;
                break;
            }
            case SQRT:
            {
                scalar_t out_val = cache_acc[k][b][n];
                scalar_t grad_out = div_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), out_val));
                int64_t child0_k = ch_acc[k][b][0];
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += grad_out;
                break;
            }
            case ADD:
            {
                int64_t child0_k = ch_acc[k][b][0];
                int64_t child1_k = ch_acc[k][b][1];
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += g_in;
#pragma omp atomic
                grad_cache_acc[child1_k][b][n] += g_in;
                break;
            }
            case SUB:
            {
                int64_t child0_k = ch_acc[k][b][0];
                int64_t child1_k = ch_acc[k][b][1];
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += g_in;
#pragma omp atomic
                grad_cache_acc[child1_k][b][n] -= g_in;
                break;
            }
            case MUL:
            {
                int64_t child0_k = ch_acc[k][b][0];
                int64_t child1_k = ch_acc[k][b][1];
                scalar_t child0_val = cache_acc[child0_k][b][n];
                scalar_t child1_val = cache_acc[child1_k][b][n];
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += mul_wrapper(g_in, child1_val);
#pragma omp atomic
                grad_cache_acc[child1_k][b][n] += mul_wrapper(g_in, child0_val);
                break;
            }
            case DIV:
            {
                int64_t child0_k = ch_acc[k][b][0];
                int64_t child1_k = ch_acc[k][b][1];
                scalar_t child0_val = cache_acc[child0_k][b][n];
                scalar_t child1_val = cache_acc[child1_k][b][n];
                scalar_t grad_out0 = div_wrapper(g_in, child1_val);
                scalar_t grad_out1 = mul_wrapper(g_in, div_wrapper(-child0_val, mul_wrapper(child1_val, child1_val)));
#pragma omp atomic
                grad_cache_acc[child0_k][b][n] += grad_out0;
#pragma omp atomic
                grad_cache_acc[child1_k][b][n] += grad_out1;
                break;
            }
            default:
            {
                if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
                {
                    size_t var_idx = op - VAR_START_ID;
#pragma omp atomic
                    grad_x_acc[n][var_idx] += g_in;
                }
            }
            }
        }
    }
}

// --- Input Validation Implementation (CPU) ---

void validate_inputs_cpu_impl(
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    int32_t *error_flag_ptr)
{
    const int64_t M = ops_acc.size(0);
    const int64_t B = ops_acc.size(1);

#pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            if (*error_flag_ptr != 0)
            {
                continue; // Another thread found an error, so we can stop.
            }

            const int op = ops_acc[m][b];

            const int expected_arity = get_arity(op);
            int actual_arity = 0;
            for (size_t i = 0; i < MAX_ARITY; ++i)
            {
                const int64_t child_k = ch_acc[m][b][i];
                if (child_k != -1)
                {
                    actual_arity++;
                    if (child_k >= m)
                    {
                        // Atomically set the error flag
                        std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 1);
                        goto next_iter;
                    }
                }
            }
            if (actual_arity != expected_arity)
            {
                // Atomically set the error flag
                std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 1);
            }
        next_iter:;
        }
    }
}

// Explicitly instantiate the templates for float and double types
template void forward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<float, 2> x_acc,
    torch::TensorAccessor<float, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void backward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 3> grad_cache_acc,
    torch::TensorAccessor<float, 1> grad_c_acc,
    torch::TensorAccessor<float, 2> grad_x_acc,
    torch::TensorAccessor<float, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void forward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<double, 2> x_acc,
    torch::TensorAccessor<double, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void backward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> grad_cache_acc,
    torch::TensorAccessor<double, 1> grad_c_acc,
    torch::TensorAccessor<double, 2> grad_x_acc,
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);