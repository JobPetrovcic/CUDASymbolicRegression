#include "evaluation_kernels.h"
#include "operators.h"
#include <omp.h>  // For CPU parallelization
#include <atomic> // For std::atomic_exchange
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
                if (child_k != NULL_CHILD)
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

// --- Forward Pass Implementation (CPU) ---
template <typename scalar_t>
void evaluation_forward_step_k_cpu_impl(
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
    for (int64_t n = 0; n < N;
         ++n)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            const int op = ops_acc[k][b];

            if (op == NO_OP)
            {
                cache_acc[k][n][b] = std::numeric_limits<scalar_t>::quiet_NaN();
                continue;
            }

            scalar_t arg0 = static_cast<scalar_t>(0.0);
            scalar_t arg1 = static_cast<scalar_t>(0.0);

            const int arity = get_arity(op);
            if (arity >= 1)
            {
                int64_t ch0_idx = ch_acc[k][b][0];
                arg0 = cache_acc[ch0_idx][n][b];
            }
            if (arity == 2)
            {
                int64_t ch1_idx = ch_acc[k][b][1];
                arg1 = cache_acc[ch1_idx][n][b];
            }

            scalar_t result = static_cast<scalar_t>(0.0);
            switch (op)
            {
            case LEARNABLE_CONSTANT:
            {
                const int64_t c_idx = posC_acc[k][b]; // Maybe TODO: add a check
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
                result = static_cast<scalar_t>(M_PI);
                break;
            case E:
                result = static_cast<scalar_t>(M_E);
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
                // This should be unreachable if validation is correct
                if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
                {
                    result = x_acc[n][op - VAR_START_ID];
                }
            }
            }
            cache_acc[k][n][b] = result;
        }
    }
}

// --- Backward Pass Implementation (CPU) ---

template <typename scalar_t>
void evaluation_backward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 3> grad_cache_acc,
    torch::TensorAccessor<scalar_t, 1> grad_c_acc,
    torch::TensorAccessor<scalar_t, 2> grad_x_acc,
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = grad_x_acc.size(0);

#pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            if (*error_flag_ptr != 0)
                continue; // Error already flagged
            scalar_t g_in = grad_cache_acc[k][n][b];

            // If the operation is NO_OP, no gradient is needed, and g_in should be zero.

            const int op = ops_acc[k][b];

            if (op == NO_OP)
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                // g_in should be 0 for NO_OP. If it's not, it's an error.
                std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 2); // Error code 2 for gradient on NO_OP
                continue;
            }

            switch (op)
            {
            case LEARNABLE_CONSTANT:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t c_idx = posC_acc[k][b];
                if (c_idx != NULL_CHILD)
                {
#pragma omp atomic
                    grad_c_acc[c_idx] += g_in;
                }
                break;
            }
            case SIN:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, cos_wrapper(arg0));
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case COS:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, -sin_wrapper(arg0));
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case EXP:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, exp_wrapper(arg0));
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case LOG:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];

                scalar_t g_out0;
                if (arg0 <= static_cast<scalar_t>(0.0))
                {
                    std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 3); // Error code 3 for gradient on invalid log
                    g_out0 = static_cast<scalar_t>(0.0);
                }
                else
                {
                    g_out0 = div_wrapper(g_in, arg0);
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case SQUARE:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), arg0));
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case SQRT:
            {
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0;
                // The commented-out code is consistent with the logic that if the g_in is zero, then the g_out0 should also be zero.
                // However, this is not consistent with pytorch's behavior, which returns NaN for gradients of sqrt at negative numbers even if g_in is zero.

                //                if (arg0 <= static_cast<scalar_t>(0.0))
                //                {
                //                    std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 4); // Error code 4 for gradient on invalid sqrt
                //                    g_out0 = static_cast<scalar_t>(0.0);
                //                }
                //                else
                //                {
                //                    g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
                //                }
                if (arg0 < static_cast<scalar_t>(0.0))
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                else
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        g_out0 = static_cast<scalar_t>(0.0); // Return zero gradient if incoming gradient is zero
                    else
                        g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case ADD:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                int64_t ch1_idx = ch_acc[k][b][1];
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_in;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] += g_in;
                break;
            }
            case SUB:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                int64_t ch1_idx = ch_acc[k][b][1];
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_in;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] -= g_in;
                break;
            }
            case MUL:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                int64_t ch1_idx = ch_acc[k][b][1];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t arg1 = cache_acc[ch1_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, arg1);
                scalar_t g_out1 = mul_wrapper(g_in, arg0);
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] += g_out1;
                break;
            }
            case DIV:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                int64_t ch1_idx = ch_acc[k][b][1];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t arg1 = cache_acc[ch1_idx][n][b];
                scalar_t g_out0, g_out1;
                if (arg1 == static_cast<scalar_t>(0.0))
                {
                    std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), 5); // Error code 5 for gradient on invalid div
                    g_out0 = static_cast<scalar_t>(0.0);
                    g_out1 = static_cast<scalar_t>(0.0);
                }
                else
                {
                    g_out0 = div_wrapper(g_in, arg1);
                    g_out1 = div_wrapper(-g_in * arg0, (arg1 * arg1));
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] += g_out1;
                break;
            }
            case POW:
            {
                int64_t ch0_idx = ch_acc[k][b][0];
                int64_t ch1_idx = ch_acc[k][b][1];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t arg1 = cache_acc[ch1_idx][n][b];
                scalar_t g_out0, g_out1;
                if (arg0 < static_cast<scalar_t>(0.0))
                {
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                    g_out1 = std::numeric_limits<scalar_t>::quiet_NaN();
                }
                else
                {
                    g_out0 = mul_wrapper(g_in, mul_wrapper(pow_wrapper(arg0, sub_wrapper(arg1, static_cast<scalar_t>(1.0))), arg1));
                    g_out1 = mul_wrapper(g_in, mul_wrapper(log_wrapper(arg0), pow_wrapper(arg0, arg1)));
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] += g_out1;
                break;
            }
            default:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
                {
                    int64_t x_idx = op - VAR_START_ID;
#pragma omp atomic
                    grad_x_acc[n][x_idx] += g_in;
                }
            }
            }
        }
    }
}

// Explicitly instantiate the templates for float and double types
template void evaluation_forward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<float, 2> x_acc,
    torch::TensorAccessor<float, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 3> grad_cache_acc,
    torch::TensorAccessor<float, 1> grad_c_acc,
    torch::TensorAccessor<float, 2> grad_x_acc,
    torch::TensorAccessor<float, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

template void evaluation_forward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<double, 2> x_acc,
    torch::TensorAccessor<double, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> grad_cache_acc,
    torch::TensorAccessor<double, 1> grad_c_acc,
    torch::TensorAccessor<double, 2> grad_x_acc,
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);