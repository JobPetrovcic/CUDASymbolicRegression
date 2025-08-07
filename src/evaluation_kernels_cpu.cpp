#include "evaluation_kernels.h"
#include "operators.h"
#include "error_codes.h"
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
                        std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::VALIDATION_CHILD_INDEX_GEQ_PARENT));
                        goto next_iter;
                    }
                }
            }
            if (actual_arity != expected_arity)
            {
                // Atomically set the error flag
                std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::VALIDATION_ARITY_MISMATCH));
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
    torch::TensorAccessor<scalar_t, 2> c_acc,
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
                result = c_acc[k][b];
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
            case TAN:
                result = tan_wrapper(arg0);
                break;
            case ARCSIN:
                result = arcsin_wrapper(arg0);
                break;
            case ARCCOS:
                result = arccos_wrapper(arg0);
                break;
            case ARCTAN:
                result = arctan_wrapper(arg0);
                break;
            case SINH:
                result = sinh_wrapper(arg0);
                break;
            case COSH:
                result = cosh_wrapper(arg0);
                break;
            case TANH:
                result = tanh_wrapper(arg0);
                break;
            case FLOOR:
                result = floor_wrapper(arg0);
                break;
            case CEIL:
                result = ceil_wrapper(arg0);
                break;
            case LN:
                result = ln_wrapper(arg0);
                break;
            case LOG10:
                result = log10_wrapper(arg0);
                break;
            case NEG:
                result = neg_wrapper(arg0);
                break;
            case INV:
                result = inv_wrapper(arg0);
                break;
            case CUBE:
                result = cube_wrapper(arg0);
                break;
            case FOURTH:
                result = fourth_wrapper(arg0);
                break;
            case FIFTH:
                result = fifth_wrapper(arg0);
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
    torch::TensorAccessor<scalar_t, 2> grad_c_acc,
    torch::TensorAccessor<scalar_t, 2> grad_x_acc,
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
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

            const int op = ops_acc[k][b];

            if (op == NO_OP)
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                // g_in should be 0 for NO_OP. If it's not, it's an error.
                std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_GRAD_ON_NO_OP)); // Error code 2 for gradient on NO_OP
                continue;
            }

            switch (op)
            {
            case LEARNABLE_CONSTANT:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
#pragma omp atomic
                grad_c_acc[k][b] += g_in;
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
                // This behavior is not consistent with pytorch
                // if (arg0 <= static_cast<scalar_t>(0.0))
                //{
                //     std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_LOG_AT_NON_POSITIVE)); // Error code 3 for gradient on invalid log
                //     g_out0 = static_cast<scalar_t>(0.0);
                // }
                // else
                //{
                //     g_out0 = div_wrapper(g_in, arg0);
                // }
                g_out0 = div_wrapper(g_in, arg0);

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
                //                    std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_SQRT_AT_NEGATIVE)); // Error code 4 for gradient on invalid sqrt
                //                    g_out0 = static_cast<scalar_t>(0.0);
                //                }
                //                else
                //                {
                //                    g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
                //                }
                if (arg0 < static_cast<scalar_t>(0.0))
                {
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                }
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
                    std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_DIV_BY_ZERO)); // Error code 5 for gradient on invalid div
                    g_out0 = static_cast<scalar_t>(0.0);
                    g_out1 = static_cast<scalar_t>(0.0);
                }
                else
                {
                    g_out0 = div_wrapper(g_in, arg1);
                    g_out1 = -div_wrapper(mul_wrapper(g_in, arg0), mul_wrapper(arg1, arg1));
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
                    if (g_in == static_cast<scalar_t>(0.0))
                    {
                        g_out0 = static_cast<scalar_t>(0.0);
                        g_out1 = static_cast<scalar_t>(0.0);
                    }
                    else
                    {
                        g_out0 = mul_wrapper(g_in, mul_wrapper(pow_wrapper(arg0, sub_wrapper(arg1, static_cast<scalar_t>(1.0))), arg1));
                        g_out1 = mul_wrapper(g_in, mul_wrapper(log_wrapper(arg0), pow_wrapper(arg0, arg1)));
                    }
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
#pragma omp atomic
                grad_cache_acc[ch1_idx][n][b] += g_out1;
                break;
            }
            case TAN:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t cos_val = cos_wrapper(arg0);
                scalar_t g_out0 = div_wrapper(g_in, mul_wrapper(cos_val, cos_val)); // sec^2(x) = 1/cos^2(x)
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case ARCSIN:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0;
                if (arg0 < static_cast<scalar_t>(-1.0) || arg0 > static_cast<scalar_t>(1.0))
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                else
                {
                    scalar_t denom = sqrt_wrapper(static_cast<scalar_t>(1.0) - mul_wrapper(arg0, arg0));
                    g_out0 = div_wrapper(g_in, denom); // d/dx arcsin(x) = 1/sqrt(1-x^2)
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case ARCCOS:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0;
                if (arg0 < static_cast<scalar_t>(-1.0) || arg0 > static_cast<scalar_t>(1.0))
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                else
                {
                    scalar_t denom = sqrt_wrapper(static_cast<scalar_t>(1.0) - mul_wrapper(arg0, arg0));
                    g_out0 = -div_wrapper(g_in, denom); // d/dx arccos(x) = -1/sqrt(1-x^2)
                }
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case ARCTAN:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t denom = static_cast<scalar_t>(1.0) + mul_wrapper(arg0, arg0);
                scalar_t g_out0 = div_wrapper(g_in, denom); // d/dx arctan(x) = 1/(1+x^2)
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case SINH:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, cosh_wrapper(arg0)); // d/dx sinh(x) = cosh(x)
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case COSH:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, sinh_wrapper(arg0)); // d/dx cosh(x) = sinh(x)
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case TANH:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t tanh_val = tanh_wrapper(arg0);
                scalar_t g_out0 = mul_wrapper(g_in, (static_cast<scalar_t>(1.0) - mul_wrapper(tanh_val, tanh_val))); // d/dx tanh(x) = 1 - tanh^2(x)
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case FLOOR:
            {
                // Floor function has zero gradient everywhere except at integer points where it's undefined
                // We'll treat it as having zero gradient everywhere for practical purposes
                break;
            }
            case CEIL:
            {
                // Ceiling function has zero gradient everywhere except at integer points where it's undefined
                // We'll treat it as having zero gradient everywhere for practical purposes
                break;
            }
            case LN:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = div_wrapper(g_in, arg0); // d/dx ln(x) = 1/x
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case LOG10:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = div_wrapper(g_in, mul_wrapper(arg0, log_wrapper(static_cast<scalar_t>(10.0)))); // d/dx log10(x) = 1/(x*ln(10))
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case NEG:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t g_out0 = -g_in; // d/dx (-x) = -1
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case INV:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0;
                if (arg0 == static_cast<scalar_t>(0.0))
                    g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                else
                    g_out0 = -div_wrapper(g_in, mul_wrapper(arg0, arg0)); // d/dx (1/x) = -1/x^2
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case CUBE:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(3.0), mul_wrapper(arg0, arg0))); // d/dx x^3 = 3*x^2
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case FOURTH:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(4.0), cube_wrapper(arg0))); // d/dx x^4 = 4*x^3
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
                break;
            }
            case FIFTH:
            {
                if (g_in == static_cast<scalar_t>(0.0))
                    continue;
                int64_t ch0_idx = ch_acc[k][b][0];
                scalar_t arg0 = cache_acc[ch0_idx][n][b];
                scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(5.0), fourth_wrapper(arg0))); // d/dx x^5 = 5*x^4
#pragma omp atomic
                grad_cache_acc[ch0_idx][n][b] += g_out0;
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
    torch::TensorAccessor<float, 2> c_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 3> grad_cache_acc,
    torch::TensorAccessor<float, 2> grad_c_acc,
    torch::TensorAccessor<float, 2> grad_x_acc,
    torch::TensorAccessor<float, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

template void evaluation_forward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<double, 2> x_acc,
    torch::TensorAccessor<double, 2> c_acc,
    int64_t n_x, int64_t k);

template void evaluation_backward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 3> grad_cache_acc,
    torch::TensorAccessor<double, 2> grad_c_acc,
    torch::TensorAccessor<double, 2> grad_x_acc,
    torch::TensorAccessor<double, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

// --- Forward Pass Implementation for Multiple Constants (CPU) ---
template <typename scalar_t>
void evaluation_multiple_forward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 4> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<scalar_t, 2> x_acc,
    torch::TensorAccessor<scalar_t, 3> c_acc,
    int64_t n_x, int64_t k, int64_t K)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = x_acc.size(0);

#pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            const int op = ops_acc[k][b];

            // Handle terminal nodes that are independent of the constant value first
            if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
            {
                scalar_t result = x_acc[n][op - VAR_START_ID];
                for (int64_t const_idx = 0; const_idx < K; ++const_idx)
                {
                    cache_acc[k][n][b][const_idx] = result;
                }
                continue; // to next (n,b) pair
            }

            // Loop over K for other ops
            for (int64_t const_idx = 0; const_idx < K; ++const_idx)
            {
                if (op == NO_OP)
                {
                    cache_acc[k][n][b][const_idx] = std::numeric_limits<scalar_t>::quiet_NaN();
                    continue;
                }

                scalar_t arg0 = static_cast<scalar_t>(0.0);
                scalar_t arg1 = static_cast<scalar_t>(0.0);

                const int arity = get_arity(op);
                if (arity >= 1)
                {
                    int64_t ch0_idx = ch_acc[k][b][0];
                    arg0 = cache_acc[ch0_idx][n][b][const_idx];
                }
                if (arity == 2)
                {
                    int64_t ch1_idx = ch_acc[k][b][1];
                    arg1 = cache_acc[ch1_idx][n][b][const_idx];
                }

                scalar_t result = static_cast<scalar_t>(0.0);
                switch (op)
                {
                case LEARNABLE_CONSTANT:
                    result = c_acc[k][b][const_idx];
                    break;
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
                case TAN:
                    result = tan_wrapper(arg0);
                    break;
                case ARCSIN:
                    result = arcsin_wrapper(arg0);
                    break;
                case ARCCOS:
                    result = arccos_wrapper(arg0);
                    break;
                case ARCTAN:
                    result = arctan_wrapper(arg0);
                    break;
                case SINH:
                    result = sinh_wrapper(arg0);
                    break;
                case COSH:
                    result = cosh_wrapper(arg0);
                    break;
                case TANH:
                    result = tanh_wrapper(arg0);
                    break;
                case FLOOR:
                    result = floor_wrapper(arg0);
                    break;
                case CEIL:
                    result = ceil_wrapper(arg0);
                    break;
                case LN:
                    result = ln_wrapper(arg0);
                    break;
                case LOG10:
                    result = log10_wrapper(arg0);
                    break;
                case NEG:
                    result = neg_wrapper(arg0);
                    break;
                case INV:
                    result = inv_wrapper(arg0);
                    break;
                case CUBE:
                    result = cube_wrapper(arg0);
                    break;
                case FOURTH:
                    result = fourth_wrapper(arg0);
                    break;
                case FIFTH:
                    result = fifth_wrapper(arg0);
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
                // VAR case is handled above the loop
                default:
                    break;
                }
                cache_acc[k][n][b][const_idx] = result;
            }
        }
    }
}

// --- Backward Pass Implementation for Multiple Constants (CPU) ---
template <typename scalar_t>
void evaluation_multiple_backward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 4> grad_cache_acc,
    torch::TensorAccessor<scalar_t, 3> grad_c_acc,
    torch::TensorAccessor<scalar_t, 2> grad_x_acc,
    torch::TensorAccessor<scalar_t, 4> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    int64_t n_x, int64_t k, int64_t K, int32_t *error_flag_ptr)
{
    const int64_t B = ops_acc.size(1);
    const int64_t N = grad_x_acc.size(0);

#pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            if (*error_flag_ptr != 0)
                continue;

            const int op = ops_acc[k][b];

            // Handle VAR case separately by summing gradients across the K dimension first
            if (op >= VAR_START_ID && op < VAR_START_ID + n_x)
            {
                scalar_t total_g_in = 0.0;
                for (int64_t const_idx = 0; const_idx < K; ++const_idx)
                {
                    total_g_in += grad_cache_acc[k][n][b][const_idx];
                }
                if (total_g_in != static_cast<scalar_t>(0.0))
                {
#pragma omp atomic
                    grad_x_acc[n][op - VAR_START_ID] += total_g_in;
                }
                continue;
            }

            // Loop over K for other ops
            for (int64_t const_idx = 0; const_idx < K; ++const_idx)
            {
                scalar_t g_in = grad_cache_acc[k][n][b][const_idx];
                if (op == NO_OP)
                {
                    if (g_in != static_cast<scalar_t>(0.0))
                        std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_GRAD_ON_NO_OP));
                    continue;
                }

                switch (op)
                {
                case LEARNABLE_CONSTANT:
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
#pragma omp atomic
                    grad_c_acc[k][b][const_idx] += g_in;
                    break;
                case SIN:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, cos_wrapper(arg0));
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case COS:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, -sin_wrapper(arg0));
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case EXP:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, exp_wrapper(arg0));
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case LOG:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0;
                    // This behavior is not consistent with pytorch
                    // if (arg0 <= 0)
                    //{
                    //     std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), //static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_LOG_AT_NON_POSITIVE));
                    //     g_out0 = static_cast<scalar_t>(0.0);
                    // }
                    // else
                    //{
                    //     g_out0 = div_wrapper(g_in, arg0);
                    // }
                    g_out0 = div_wrapper(g_in, arg0);

#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case SQUARE:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), arg0));
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case SQRT:
                {
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0;
                    if (arg0 < 0)
                    {
                        g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                    }
                    else
                    {
                        if (g_in == static_cast<scalar_t>(0.0))
                            g_out0 = static_cast<scalar_t>(0.0);
                        else
                            g_out0 = div_wrapper(g_in * static_cast<scalar_t>(0.5), sqrt_wrapper(arg0));
                    }
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case ADD:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    int64_t ch1 = ch_acc[k][b][1];
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_in;
#pragma omp atomic
                    grad_cache_acc[ch1][n][b][const_idx] += g_in;
                    break;
                }
                case SUB:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    int64_t ch1 = ch_acc[k][b][1];
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_in;
#pragma omp atomic
                    grad_cache_acc[ch1][n][b][const_idx] -= g_in;
                    break;
                }
                case MUL:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    int64_t ch1 = ch_acc[k][b][1];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t arg1 = cache_acc[ch1][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, arg1);
                    scalar_t g_out1 = mul_wrapper(g_in, arg0);
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
#pragma omp atomic
                    grad_cache_acc[ch1][n][b][const_idx] += g_out1;
                    break;
                }
                case DIV:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    int64_t ch1 = ch_acc[k][b][1];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t arg1 = cache_acc[ch1][n][b][const_idx];
                    scalar_t g_out0, g_out1;
                    if (arg1 == 0)
                    {
                        std::atomic_exchange(reinterpret_cast<std::atomic<int32_t> *>(error_flag_ptr), static_cast<int32_t>(ErrorCode::EVAL_BACKWARD_DIV_BY_ZERO));
                        g_out0 = static_cast<scalar_t>(0.0);
                        g_out1 = static_cast<scalar_t>(0.0);
                    }
                    else
                    {
                        g_out0 = div_wrapper(g_in, arg1);
                        g_out1 = -div_wrapper(mul_wrapper(g_in, arg0), mul_wrapper(arg1, arg1));
                    }
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
#pragma omp atomic
                    grad_cache_acc[ch1][n][b][const_idx] += g_out1;
                    break;
                }
                case POW:
                {
                    int64_t ch0 = ch_acc[k][b][0];
                    int64_t ch1 = ch_acc[k][b][1];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t arg1 = cache_acc[ch1][n][b][const_idx];
                    scalar_t g_out0, g_out1;
                    if (arg0 < 0)
                    {
                        g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                        g_out1 = std::numeric_limits<scalar_t>::quiet_NaN();
                    }
                    else
                    {
                        if (g_in == static_cast<scalar_t>(0.0))
                        {
                            g_out0 = static_cast<scalar_t>(0.0);
                            g_out1 = static_cast<scalar_t>(0.0);
                        }
                        else
                        {
                            g_out0 = mul_wrapper(g_in, mul_wrapper(pow_wrapper(arg0, sub_wrapper(arg1, static_cast<scalar_t>(1.0))), arg1));
                            g_out1 = mul_wrapper(g_in, mul_wrapper(log_wrapper(arg0), pow_wrapper(arg0, arg1)));
                        }
                    }
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
#pragma omp atomic
                    grad_cache_acc[ch1][n][b][const_idx] += g_out1;
                    break;
                }
                case TAN:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t cos_val = cos_wrapper(arg0);
                    scalar_t g_out0 = div_wrapper(g_in, mul_wrapper(cos_val, cos_val)); // sec^2(x) = 1/cos^2(x)
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case ARCSIN:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0;
                    if (arg0 < static_cast<scalar_t>(-1.0) || arg0 > static_cast<scalar_t>(1.0))
                        g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                    else
                    {
                        scalar_t denom = sqrt_wrapper(static_cast<scalar_t>(1.0) - mul_wrapper(arg0, arg0));
                        g_out0 = div_wrapper(g_in, denom); // d/dx arcsin(x) = 1/sqrt(1-x^2)
                    }
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case ARCCOS:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0;
                    if (arg0 < static_cast<scalar_t>(-1.0) || arg0 > static_cast<scalar_t>(1.0))
                        g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                    else
                    {
                        scalar_t denom = sqrt_wrapper(static_cast<scalar_t>(1.0) - mul_wrapper(arg0, arg0));
                        g_out0 = -div_wrapper(g_in, denom); // d/dx arccos(x) = -1/sqrt(1-x^2)
                    }
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case ARCTAN:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t denom = static_cast<scalar_t>(1.0) + mul_wrapper(arg0, arg0);
                    scalar_t g_out0 = div_wrapper(g_in, denom); // d/dx arctan(x) = 1/(1+x^2)
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case SINH:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, cosh_wrapper(arg0)); // d/dx sinh(x) = cosh(x)
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case COSH:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, sinh_wrapper(arg0)); // d/dx cosh(x) = sinh(x)
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case TANH:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t tanh_val = tanh_wrapper(arg0);
                    scalar_t g_out0 = mul_wrapper(g_in, (static_cast<scalar_t>(1.0) - mul_wrapper(tanh_val, tanh_val))); // d/dx tanh(x) = 1 - tanh^2(x)
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case FLOOR:
                {
                    // Floor function has zero gradient everywhere except at integer points where it's undefined
                    // We'll treat it as having zero gradient everywhere for practical purposes
                    break;
                }
                case CEIL:
                {
                    // Ceiling function has zero gradient everywhere except at integer points where it's undefined
                    // We'll treat it as having zero gradient everywhere for practical purposes
                    break;
                }
                case LN:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = div_wrapper(g_in, arg0); // d/dx ln(x) = 1/x
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case LOG10:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = div_wrapper(g_in, mul_wrapper(arg0, log_wrapper(static_cast<scalar_t>(10.0)))); // d/dx log10(x) = 1/(x*ln(10))
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case NEG:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t g_out0 = -g_in; // d/dx (-x) = -1
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case INV:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0;
                    if (arg0 == static_cast<scalar_t>(0.0))
                        g_out0 = std::numeric_limits<scalar_t>::quiet_NaN();
                    else
                        g_out0 = -div_wrapper(g_in, mul_wrapper(arg0, arg0)); // d/dx (1/x) = -1/x^2
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case CUBE:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(3.0), mul_wrapper(arg0, arg0))); // d/dx x^3 = 3*x^2
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case FOURTH:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(4.0), cube_wrapper(arg0))); // d/dx x^4 = 4*x^3
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                case FIFTH:
                {
                    if (g_in == static_cast<scalar_t>(0.0))
                        continue;
                    int64_t ch0 = ch_acc[k][b][0];
                    scalar_t arg0 = cache_acc[ch0][n][b][const_idx];
                    scalar_t g_out0 = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(5.0), fourth_wrapper(arg0))); // d/dx x^5 = 5*x^4
#pragma omp atomic
                    grad_cache_acc[ch0][n][b][const_idx] += g_out0;
                    break;
                }
                default:
                    // VAR case is handled above the loop
                    break;
                }
            }
        }
    }
}

// Explicitly instantiate the templates for float and double types
template void evaluation_multiple_forward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 4> cache_acc, torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc, torch::TensorAccessor<float, 2> x_acc,
    torch::TensorAccessor<float, 3> c_acc, int64_t n_x, int64_t k, int64_t K);
template void evaluation_multiple_backward_step_k_cpu_impl<float>(
    torch::TensorAccessor<float, 4> grad_cache_acc, torch::TensorAccessor<float, 3> grad_c_acc,
    torch::TensorAccessor<float, 2> grad_x_acc, torch::TensorAccessor<float, 4> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc, torch::TensorAccessor<int64_t, 3> ch_acc,
    int64_t n_x, int64_t k, int64_t K, int32_t *error_flag_ptr);

template void evaluation_multiple_forward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 4> cache_acc, torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc, torch::TensorAccessor<double, 2> x_acc,
    torch::TensorAccessor<double, 3> c_acc, int64_t n_x, int64_t k, int64_t K);
template void evaluation_multiple_backward_step_k_cpu_impl<double>(
    torch::TensorAccessor<double, 4> grad_cache_acc, torch::TensorAccessor<double, 3> grad_c_acc,
    torch::TensorAccessor<double, 2> grad_x_acc, torch::TensorAccessor<double, 4> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc, torch::TensorAccessor<int64_t, 3> ch_acc,
    int64_t n_x, int64_t k, int64_t K, int32_t *error_flag_ptr);