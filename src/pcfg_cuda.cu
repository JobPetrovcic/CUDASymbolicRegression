#include "pcfg_kernels.h"
#include "operators.h"
#include <torch/torch.h>
#include <cuda.h>
#include <curand_kernel.h>

constexpr int THREAD_COUNT = 256;

__global__ void pcfg_sample_string_expression_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> rule_lhs,
    const torch::PackedTensorAccessor32<int64_t, 1> rhs_ptr,
    const torch::PackedTensorAccessor32<int64_t, 1> rhs_concat,
    const torch::PackedTensorAccessor32<int64_t, 1> nt_rule_ptr,
    const torch::PackedTensorAccessor32<float, 1> nt_rule_cum_probs,
    int64_t start_symbol_id,
    int64_t max_length,
    int64_t max_tries,
    torch::PackedTensorAccessor32<int64_t, 2> output,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    const torch::PackedTensorAccessor32<int64_t, 1> seeds,
    int64_t B)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors[b] = 0;

    curandState_t state;
    curand_init(seeds[b], 0, 0, &state);

    bool generated_successfully = false;
    for (int32_t t = 0; t < max_tries; ++t)
    {
        bool should_restart = false;
        int32_t stack[HARD_MAX_LENGTH];
        int32_t stack_ptr = 0;
        stack[stack_ptr++] = start_symbol_id;

        int32_t out_ptr = 0;

        while (stack_ptr > 0)
        {
            int32_t current_symbol = stack[--stack_ptr];
            const int32_t rule_start = nt_rule_ptr[current_symbol];
            const int32_t rule_end = nt_rule_ptr[current_symbol + 1];
            if (rule_start != rule_end)
            { // Non-terminal
                float rand_val = curand_uniform(&state);
                int32_t chosen_rule = INVALID_RULE;
                for (int32_t r = rule_start; r < rule_end; ++r)
                {
                    if (rand_val <= nt_rule_cum_probs[r])
                    {
                        chosen_rule = r;
                        break;
                    }
                }
                if (chosen_rule == INVALID_RULE)
                {
                    errors[b] = 10; // No valid rule found
                    break;
                }

                int32_t rhs_start = rhs_ptr[chosen_rule];
                int32_t rhs_end = rhs_ptr[chosen_rule + 1];
                for (int32_t i = rhs_end - 1; i >= rhs_start; --i)
                {
                    if (stack_ptr >= max_length)
                    {
                        should_restart = true; // Expression too long
                        break;
                    }
                    stack[stack_ptr++] = rhs_concat[i];
                }
                if (should_restart)
                {
                    break; // Restart the generation for this batch item
                }
            }
            else
            { // Terminal
                if (out_ptr >= max_length)
                {
                    should_restart = true; // Expression too long
                    break;
                }
                output[b][out_ptr++] = current_symbol;
            }
        }

        if (should_restart)
        {
            continue;
        }

        if (errors[b] != 0)
            return;

        if (stack_ptr == 0 && out_ptr > 0)
        {
            // Successfully generated an expression
            for (int32_t i = out_ptr; i < max_length; ++i)
            {
                output[b][i] = NO_OP; // padding
            }
            generated_successfully = true;
            break;
        }
    }

    if (!generated_successfully)
    {
        errors[b] = 9; // Using a new error code to avoid conflict
    }
}

void pcfg_sample_string_expression_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> rule_lhs,
    const torch::PackedTensorAccessor32<int64_t, 1> rhs_ptr,
    const torch::PackedTensorAccessor32<int64_t, 1> rhs_concat,
    const torch::PackedTensorAccessor32<int64_t, 1> nt_rule_ptr,
    const torch::PackedTensorAccessor32<float, 1> nt_rule_cum_probs,
    int64_t start_symbol_id,
    int64_t max_length,
    int64_t max_tries,
    torch::PackedTensorAccessor32<int64_t, 2> output,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    torch::PackedTensorAccessor32<int64_t, 1> seeds,
    int64_t B)
{
    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;
    pcfg_sample_string_expression_kernel<<<blocks, threads>>>(
        rule_lhs,
        rhs_ptr,
        rhs_concat,
        nt_rule_ptr,
        nt_rule_cum_probs,
        start_symbol_id,
        max_length,
        max_tries,
        output,
        errors,
        seeds,
        B);
}

__global__ void parse_to_postfix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 3> children,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors[b] = 0;
    int32_t op_stack[HARD_MAX_LENGTH];
    int32_t op_stack_ptr = 0;
    int32_t out_queue[HARD_MAX_LENGTH];
    int32_t out_queue_size = 0;

    for (int32_t i = 0; i < M && expressions[b][i] != NO_OP; ++i)
    {
        const int32_t token = expressions[b][i];
        if (precedence[token] == 0)
        { // Terminal
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = 1; // Postfix expression too long
                return;
            }
            out_queue[out_queue_size++] = token;
        }
        else if (token == lparenthesis_id)
        {
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = 2; // Operator stack overflow
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
        else if (token == rparenthesis_id)
        {
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = 1; // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr == 0)
            {
                errors[b] = 3; // Mismatched parenthesis
                return;
            }
            op_stack_ptr--; // Pop lparenthesis
        }
        else
        { // Operator
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                   ((!is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token]) ||
                    (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token])))
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = 1; // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = 2; // Operator stack overflow
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
    }

    while (op_stack_ptr > 0)
    {
        if (op_stack[op_stack_ptr - 1] == lparenthesis_id)
        {
            errors[b] = 3; // Mismatched parenthesis
            return;
        }
        if (out_queue_size >= HARD_MAX_LENGTH)
        {
            errors[b] = 1; // Postfix expression too long
            return;
        }
        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
    }

    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        ops[b][i] = out_queue[i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        ops[b][i] = NO_OP; // padding
    }

    int32_t child_stack[HARD_MAX_LENGTH];
    int32_t child_stack_ptr = 0;
    for (int32_t i = 0; i < out_queue_size; i++)
    {
        int32_t token = out_queue[i];
        if (is_unary(token))
        {
            if (child_stack_ptr < 1)
            {
                errors[b] = 5; // Unary operator without operand
                return;
            }
            int32_t child_index = child_stack[--child_stack_ptr];
            children[b][i][0] = child_index;
            children[b][i][1] = NULL_CHILD;
        }
        else if (is_binary(token))
        {
            if (child_stack_ptr < 2)
            {
                errors[b] = 6; // Binary operator without enough operands
                return;
            }
            int32_t right_child_index = child_stack[--child_stack_ptr];
            int32_t left_child_index = child_stack[--child_stack_ptr];
            children[b][i][0] = left_child_index;
            children[b][i][1] = right_child_index;
        }
        else
        {
            // no children
            children[b][i][0] = NULL_CHILD;
            children[b][i][1] = NULL_CHILD;
        }
        if (child_stack_ptr >= HARD_MAX_LENGTH)
        {
            errors[b] = 7; // Child stack overflow
            return;
        }
        child_stack[child_stack_ptr++] = i;
    }
    if (out_queue_size > 0 && child_stack_ptr != 1)
        errors[b] = 8; // Malformed expression (e.g., too many operands)
}

void parse_to_postfix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 3> children,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    parse_to_postfix_kernel<<<blocks, threads>>>(
        precedence,
        expressions,
        ops,
        children,
        errors,
        lparenthesis_id,
        rparenthesis_id,
        B,
        M);
}
