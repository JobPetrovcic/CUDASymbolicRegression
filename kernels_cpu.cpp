#include "kernels.h"
#include "symbolic_evaluation.h"
#include <omp.h> // For CPU parallelization

// --- Forward Pass Implementation (CPU) ---

void forward_step_k_impl(
    torch::Tensor cache, const torch::Tensor &ops, const torch::Tensor &ch,
    const torch::Tensor &x, const torch::Tensor &Constants,
    const torch::Tensor &ConstantPosition, int64_t k)
{

    // Dispatch to the correct C++ type (float, double, half) based on the tensor's dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(), "forward_step_k_cpu", ([&]
                                                                                    {
        // Get data pointers for all tensors
        auto cache_ptr = cache.data_ptr<scalar_t>();
        auto ops_ptr = ops.data_ptr<int64_t>();
        auto ch_ptr = ch.data_ptr<int64_t>();
        auto x_ptr = x.data_ptr<scalar_t>();
        auto c_ptr = Constants.data_ptr<scalar_t>();
        auto ConstantPosition_ptr = ConstantPosition.data_ptr<int64_t>();

        // Get dimensions
        const auto M = ops.size(0);
        const auto B = ops.size(1);
        const auto N = x.size(0);
        const auto n_x = x.size(1);

// Parallelize the loops over the batch (B) and data (N) dimensions
#pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const size_t op_idx = k * B + b;
                const int op = ops_ptr[op_idx];

                // Skip NO_OP entirely, leaving the cache value unchanged.
                if (op == NO_OP) {
                    continue;
                }

                scalar_t arg0 = static_cast<scalar_t>(0.0);
                scalar_t arg1 = static_cast<scalar_t>(0.0);

                const int arity = get_arity(op);
                if (arity >= 1) {
                    size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                    arg0 = cache_ptr[child0_k * B * N + b * N + n];
                }
                if (arity == 2) {
                    size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
                    arg1 = cache_ptr[child1_k * B * N + b * N + n];
                }
                
                scalar_t result = static_cast<scalar_t>(0.0);
                switch (op) {
                    case LEARNABLE_CONSTANT: {
                        int64_t c_idx = ConstantPosition_ptr[op_idx];
                        if (c_idx != -1) result = c_ptr[c_idx];
                        break;
                    }
                    case CONST_1: result = static_cast<scalar_t>(1.0); break;
                    case SIN: result = sin_wrapper(arg0); break;
                    case COS: result = cos_wrapper(arg0); break;
                    case EXP: result = exp_wrapper(arg0); break;
                    case LOG: result = log_wrapper(arg0); break;
                    case SQUARE: result = square_wrapper(arg0); break;
                    case SQRT: result = sqrt_wrapper(arg0); break;
                    case ADD: result = add_wrapper(arg0, arg1); break;
                    case SUB: result = sub_wrapper(arg0, arg1); break;
                    case MUL: result = mul_wrapper(arg0, arg1); break;
                    case DIV: result = div_wrapper(arg0, arg1); break;
                    default: {
                        if (op >= VAR_START_ID && op < VAR_START_ID + n_x) {
                            size_t var_idx = op - VAR_START_ID;
                            result = x_ptr[n * n_x + var_idx];
                        }
                    }
                }
                cache_ptr[k * B * N + b * N + n] = result;
            }
        } }));
}

// --- Backward Pass Implementation (CPU) ---

void backward_step_k_impl(
    torch::Tensor grad_cache, torch::Tensor grad_Constants, torch::Tensor grad_x,
    const torch::Tensor &cache, const torch::Tensor &ops, const torch::Tensor &ch,
    const torch::Tensor &ConstantPosition, int64_t k)
{

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_cache.scalar_type(), "backward_step_k_cpu", ([&]
                                                                                          {
        // Get data pointers
        auto grad_cache_ptr = grad_cache.data_ptr<scalar_t>();
        auto grad_c_ptr = grad_Constants.data_ptr<scalar_t>();
        auto grad_x_ptr = grad_x.data_ptr<scalar_t>();
        auto cache_ptr = cache.data_ptr<scalar_t>();
        auto ops_ptr = ops.data_ptr<int64_t>();
        auto ch_ptr = ch.data_ptr<int64_t>();
        auto ConstantPosition_ptr = ConstantPosition.data_ptr<int64_t>();

        // Get dimensions
        const auto B = ops.size(1);
        const auto N = cache.size(2);
        const auto n_x = grad_x.size(1);

#pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const size_t current_idx = k * B * N + b * N + n;
                const scalar_t g_in = grad_cache_ptr[current_idx];
                
                // Optimization: if incoming gradient is zero, no work to do.
                if (g_in == static_cast<scalar_t>(0.0)) {
                    continue;
                }

                const size_t op_idx = k * B + b;
                const int op = ops_ptr[op_idx];

                switch (op) {
                    case LEARNABLE_CONSTANT: {
                        int64_t c_idx = ConstantPosition_ptr[op_idx];
                        if (c_idx != -1) {
#pragma omp atomic
                            grad_c_ptr[c_idx] += g_in;
                        }
                        break;
                    }
                    case SIN: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        scalar_t child_val = cache_ptr[child0_k * B * N + b * N + n];
                        scalar_t grad_out = mul_wrapper(g_in, cos_wrapper(child_val));
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += grad_out;
                        break;
                    }
                    case COS: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        scalar_t child_val = cache_ptr[child0_k * B * N + b * N + n];
                        scalar_t grad_out = mul_wrapper(g_in, -cos_wrapper(child_val));
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += grad_out;
                        break;
                    }
                    case SQUARE: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        scalar_t child_val = cache_ptr[child0_k * B * N + b * N + n];
                        scalar_t grad_out = mul_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), child_val));
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += grad_out;
                        break;
                    }
                    case SQRT: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        scalar_t out_val = cache_ptr[current_idx];
                        scalar_t grad_out = div_wrapper(g_in, mul_wrapper(static_cast<scalar_t>(2.0), out_val));
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += grad_out;
                        break;
                    }
                    case ADD: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += g_in;
#pragma omp atomic
                        grad_cache_ptr[child1_k * B * N + b * N + n] += g_in;
                        break;
                    }
                    case SUB: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += g_in;
#pragma omp atomic
                        grad_cache_ptr[child1_k * B * N + b * N + n] -= g_in;
                        break;
                    }
                    case MUL: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
                        scalar_t child0_val = cache_ptr[child0_k * B * N + b * N + n];
                        scalar_t child1_val = cache_ptr[child1_k * B * N + b * N + n];
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += mul_wrapper(g_in, child1_val);
#pragma omp atomic
                        grad_cache_ptr[child1_k * B * N + b * N + n] += mul_wrapper(g_in, child0_val);
                        break;
                    }
                    case DIV: {
                        size_t child0_k = ch_ptr[op_idx * MAX_ARITY + 0];
                        size_t child1_k = ch_ptr[op_idx * MAX_ARITY + 1];
                        scalar_t child0_val = cache_ptr[child0_k * B * N + b * N + n];
                        scalar_t child1_val = cache_ptr[child1_k * B * N + b * N + n];
#pragma omp atomic
                        grad_cache_ptr[child0_k * B * N + b * N + n] += div_wrapper(g_in, child1_val);
#pragma omp atomic
                        grad_cache_ptr[child1_k * B * N + b * N + n] -= mul_wrapper(g_in, div_wrapper(child0_val, square_wrapper(child1_val)));
                        break;
                    }
                    default: {
                        if (op >= VAR_START_ID && op < VAR_START_ID + n_x) {
                            size_t var_idx = op - VAR_START_ID;
#pragma omp atomic
                            grad_x_ptr[n * n_x + var_idx] += g_in;
                        }
                    }
                }
            }
        } }));
}

// --- Validation Implementation (CPU) ---

void validate_inputs_impl(const torch::Tensor &ops, const torch::Tensor &ch, torch::Tensor &error_flag)
{
    auto ops_ptr = ops.data_ptr<int64_t>();
    auto ch_ptr = ch.data_ptr<int64_t>();
    auto error_flag_ptr = error_flag.data_ptr<int32_t>();

    // Reset flag before checking
    *error_flag_ptr = 0;

    const auto M = ops.size(0);
    const auto B = ops.size(1);

#pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m)
    {
        for (int64_t b = 0; b < B; ++b)
        {
            // Early exit if another thread has already found an error
            if (*error_flag_ptr != 0)
            {
                continue;
            }

            const size_t op_idx = m * B + b;
            const int op = ops_ptr[op_idx];

            const int expected_arity = get_arity(op);
            int actual_arity = 0;
            for (int i = 0; i < MAX_ARITY; ++i)
            {
                int64_t child_k = ch_ptr[op_idx * MAX_ARITY + i];
                if (child_k != -1)
                {
                    actual_arity++;
                    if (child_k >= m)
                    {
                        *error_flag_ptr = 1; // Error: Invalid child index (DAG violation)
                    }
                }
            }
            if (actual_arity != expected_arity)
            {
                *error_flag_ptr = 2; // Error: Arity mismatch
            }
        }
    }
}