#include "pcfg_kernels.h"
#include "operators.h"
#include "error_codes.h"
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
                    errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_NO_VALID_RULE_FOUND); // No valid rule found
                    break;
                }

                int32_t rhs_start = rhs_ptr[chosen_rule];
                int32_t rhs_end = rhs_ptr[chosen_rule + 1];
                for (int32_t i = rhs_end - 1; i >= rhs_start; --i)
                {
                    if (stack_ptr >= HARD_MAX_LENGTH)
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
            errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);
            break;
        }
    }

    if (!generated_successfully && errors[b] == 0)
    {
        errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_MAX_TRIES_EXCEEDED); // Using a new error code to avoid conflict
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

__global__ void parse_to_prefix_kernel(
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

    int32_t expression_len = 0;
    while (expression_len < M && expressions[b][expression_len] != NO_OP)
    {
        expression_len++;
    }

    for (int32_t i = expression_len - 1; i >= 0; --i)
    {
        const int32_t token = expressions[b][i];
        if (precedence[token] == 0)
        { // Terminal
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                return;
            }
            out_queue[out_queue_size++] = token;
        }
        else if (token == rparenthesis_id) // Treat as lparenthesis
        {
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
        else if (token == lparenthesis_id) // Treat as rparenthesis
        {
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr == 0)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                return;
            }
            op_stack_ptr--; // Pop rparenthesis
        }
        else
        { // Operator
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id &&
                   ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token]) ||
                    (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token])))
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
    }

    while (op_stack_ptr > 0)
    {
        if (op_stack[op_stack_ptr - 1] == rparenthesis_id)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
            return;
        }
        if (out_queue_size >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
            return;
        }
        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
    }

    // Reverse the output queue to get prefix
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        ops[b][i] = out_queue[out_queue_size - 1 - i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        ops[b][i] = NO_OP; // padding
    }

    // Re-use out_queue to store the prefix expression for child calculation
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        out_queue[i] = ops[b][i];
    }

    int32_t child_stack[HARD_MAX_LENGTH];
    int32_t child_stack_ptr = 0;
    for (int32_t i = out_queue_size - 1; i >= 0; i--)
    {
        int32_t token = out_queue[i];
        if (is_unary(token))
        {
            if (child_stack_ptr < 1)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND); // Unary operator without operand
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS); // Binary operator without enough operands
                return;
            }
            int32_t left_child_index = child_stack[--child_stack_ptr];
            int32_t right_child_index = child_stack[--child_stack_ptr];
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
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW); // Child stack overflow
            return;
        }
        child_stack[child_stack_ptr++] = i;
    }
    if (out_queue_size > 0 && child_stack_ptr != 1 && errors[b] == 0)
        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression (e.g., too many operands)
}

void parse_to_prefix_cuda_impl(
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
    parse_to_prefix_kernel<<<blocks, threads>>>(
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                return;
            }
            out_queue[out_queue_size++] = token;
        }
        else if (token == lparenthesis_id)
        {
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
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
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr == 0)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                return;
            }
            op_stack_ptr--; // Pop lparenthesis
        }
        else
        { // Operator
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                   ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token]) ||
                    (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token])))
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
    }

    while (op_stack_ptr > 0)
    {
        if (op_stack[op_stack_ptr - 1] == lparenthesis_id)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
            return;
        }
        if (out_queue_size >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND); // Unary operator without operand
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS); // Binary operator without enough operands
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
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW); // Child stack overflow
            return;
        }
        child_stack[child_stack_ptr++] = i;
    }
    if (out_queue_size > 0 && child_stack_ptr != 1 && errors[b] == 0)
        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression (e.g., too many operands)
}

__global__ void parse_to_prefix_parent_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
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
    int32_t expression_len = 0;
    while (expression_len < M && expressions[b][expression_len] != NO_OP)
    {
        expression_len++;
    }
    for (int32_t i = expression_len - 1; i >= 0; --i)
    {
        const int32_t token = expressions[b][i];
        if (precedence[token] == 0)
        { // Terminal
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                return;
            }
            out_queue[out_queue_size++] = token;
        }
        else if (token == rparenthesis_id) // Treat as lparenthesis
        {
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
        else if (token == lparenthesis_id) // Treat as rparenthesis
        {
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr == 0)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                return;
            }
            op_stack_ptr--; // Pop rparenthesis
        }
        else
        { // Operator
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id &&
                   ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token]) ||
                    (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token])))
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
    }
    while (op_stack_ptr > 0)
    {
        if (op_stack[op_stack_ptr - 1] == rparenthesis_id)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
            return;
        }
        if (out_queue_size >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
            return;
        }
        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
    }
    // Reverse the output queue to get prefix and write to ops tensor
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        ops[b][i] = out_queue[out_queue_size - 1 - i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        ops[b][i] = NO_OP; // padding
    }
    // Initialize parents tensor
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        parents[b][i] = NULL_PARENT;
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        parents[b][i] = NO_OP; // padding
    }
    if (out_queue_size == 0)
    {
        return;
    }
    // Re-use out_queue to store the prefix expression for parent calculation
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        out_queue[i] = ops[b][i];
    }
    int32_t node_stack[HARD_MAX_LENGTH];
    int32_t node_stack_ptr = 0;
    for (int32_t i = out_queue_size - 1; i >= 0; i--)
    {
        int32_t token = out_queue[i];
        if (is_unary(token))
        {
            if (node_stack_ptr < 1)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                return;
            }
            int32_t child_index = node_stack[--node_stack_ptr];
            parents[b][child_index] = i;
        }
        else if (is_binary(token))
        {
            if (node_stack_ptr < 2)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                return;
            }
            int32_t child1_index = node_stack[--node_stack_ptr];
            int32_t child2_index = node_stack[--node_stack_ptr];
            parents[b][child1_index] = i;
            parents[b][child2_index] = i;
        }
        if (node_stack_ptr >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
            return;
        }
        node_stack[node_stack_ptr++] = i;
    }
    if (out_queue_size > 0 && node_stack_ptr != 1)
    {
        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
    }
}
void parse_to_prefix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    parse_to_prefix_parent_kernel<<<blocks, threads>>>(
        precedence,
        expressions,
        ops,
        parents,
        errors,
        lparenthesis_id,
        rparenthesis_id,
        B,
        M);
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

__global__ void parse_to_postfix_parent_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                return;
            }
            out_queue[out_queue_size++] = token;
        }
        else if (token == lparenthesis_id)
        {
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
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
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr == 0)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                return;
            }
            op_stack_ptr--; // Pop lparenthesis
        }
        else
        { // Operator
            while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                   ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token]) ||
                    (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token])))
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    return;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (op_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                return;
            }
            op_stack[op_stack_ptr++] = token;
        }
    }

    while (op_stack_ptr > 0)
    {
        if (op_stack[op_stack_ptr - 1] == lparenthesis_id)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
            return;
        }
        if (out_queue_size >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
            return;
        }
        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
    }

    // Write postfix expression to ops tensor
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        ops[b][i] = out_queue[i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        ops[b][i] = NO_OP; // padding
    }

    // Initialize parents tensor
    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        parents[b][i] = NULL_PARENT;
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        parents[b][i] = NO_OP; // padding
    }

    if (out_queue_size == 0)
    {
        return;
    }

    int32_t node_stack[HARD_MAX_LENGTH];
    int32_t node_stack_ptr = 0;
    for (int32_t i = 0; i < out_queue_size; i++)
    {
        int32_t token = out_queue[i];
        if (is_unary(token))
        {
            if (node_stack_ptr < 1)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                return;
            }
            int32_t child_index = node_stack[--node_stack_ptr];
            parents[b][child_index] = i;
        }
        else if (is_binary(token))
        {
            if (node_stack_ptr < 2)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                return;
            }
            int32_t right_child_index = node_stack[--node_stack_ptr];
            int32_t left_child_index = node_stack[--node_stack_ptr];
            parents[b][right_child_index] = i;
            parents[b][left_child_index] = i;
        }

        if (node_stack_ptr >= HARD_MAX_LENGTH)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
            return;
        }
        node_stack[node_stack_ptr++] = i;
    }
    if (out_queue_size > 0 && node_stack_ptr != 1 && errors[b] == 0)
        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression
}

void parse_to_postfix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    parse_to_postfix_parent_kernel<<<blocks, threads>>>(
        precedence,
        expressions,
        ops,
        parents,
        errors,
        lparenthesis_id,
        rparenthesis_id,
        B,
        M);
}

__global__ void postfix_to_infix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = 0;

    // Per-thread workspace for building sub-expressions
    int64_t workspace[HARD_MAX_LENGTH * 2];
    int32_t workspace_ptr = 0;

    // Stack holds pointers (start_idx, len) into the workspace
    int32_t stack_starts[HARD_MAX_LENGTH * 2];
    int32_t stack_lens[HARD_MAX_LENGTH * 2];
    int32_t stack_ptr = 0;

    for (int j = 0; j < M_postfix; ++j)
    {
        int64_t token_id = postfix_acc[b][j];
        if (token_id == NO_OP)
            break; // MAYBE TODO: continue?

        int arity = get_arity(token_id);

        if (arity == 0)
        { // Operand
            if (stack_ptr >= HARD_MAX_LENGTH * 2 || workspace_ptr >= HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = 1;
            workspace[workspace_ptr++] = token_id;
            stack_ptr++;
        }
        else if (arity == 1)
        { // Unary operator
            if (stack_ptr < 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
                break;
            }
            stack_ptr--;
            int32_t op_start = stack_starts[stack_ptr];
            int32_t op_len = stack_lens[stack_ptr];

            // Functional style: op(operand)
            if (is_functional_style(token_id))
            {
                if (workspace_ptr + 3 + op_len > HARD_MAX_LENGTH * 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    break;
                }

                stack_starts[stack_ptr] = workspace_ptr;
                stack_lens[stack_ptr] = 3 + op_len; // FIX
                workspace[workspace_ptr++] = token_id;

                workspace[workspace_ptr++] = lparen_id;
                for (int k = 0; k < op_len; ++k)
                    workspace[workspace_ptr++] = workspace[op_start + k];
                workspace[workspace_ptr++] = rparen_id;
            }
            else
            { // Postfix style: (operand)^2
                if (workspace_ptr + 3 + op_len > HARD_MAX_LENGTH * 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    break;
                }
                stack_starts[stack_ptr] = workspace_ptr;
                stack_lens[stack_ptr] = 3 + op_len;
                workspace[workspace_ptr++] = lparen_id;
                for (int k = 0; k < op_len; ++k)
                    workspace[workspace_ptr++] = workspace[op_start + k];
                workspace[workspace_ptr++] = rparen_id;
                workspace[workspace_ptr++] = token_id;
            }
            stack_ptr++;
        }
        else if (arity == 2)
        { // Binary operator
            if (stack_ptr < 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }
            stack_ptr--;
            int32_t op2_start = stack_starts[stack_ptr];
            int32_t op2_len = stack_lens[stack_ptr];
            stack_ptr--;
            int32_t op1_start = stack_starts[stack_ptr];
            int32_t op1_len = stack_lens[stack_ptr];

            if (workspace_ptr + 3 + op1_len + op2_len > HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = 3 + op1_len + op2_len;
            workspace[workspace_ptr++] = lparen_id;
            for (int k = 0; k < op1_len; ++k)
                workspace[workspace_ptr++] = workspace[op1_start + k];
            workspace[workspace_ptr++] = token_id;
            for (int k = 0; k < op2_len; ++k)
                workspace[workspace_ptr++] = workspace[op2_start + k];
            workspace[workspace_ptr++] = rparen_id;
            stack_ptr++;
        }
    }

    if (errors_acc[b] == 0)
    {
        if (stack_ptr != 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
        }
        else
        {
            int32_t final_len = stack_lens[0];
            if (final_len > M_infix)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
            }
            else
            {
                int32_t final_start = stack_starts[0];
                for (int k = 0; k < final_len; ++k)
                {
                    infix_acc[b][k] = workspace[final_start + k];
                }
            }
        }
    }
}

void postfix_to_infix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    postfix_to_infix_kernel<<<blocks, threads>>>(
        postfix_acc, infix_acc, errors_acc, lparen_id, rparen_id, B, M_postfix, M_infix);
}

__global__ void prefix_to_infix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_prefix, int64_t M_infix)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = 0;

    // Per-thread workspace for building sub-expressions
    int64_t workspace[HARD_MAX_LENGTH * 2];
    int32_t workspace_ptr = 0;

    // Stack holds pointers (start_idx, len) into the workspace
    int32_t stack_starts[HARD_MAX_LENGTH * 2];
    int32_t stack_lens[HARD_MAX_LENGTH * 2];
    int32_t stack_ptr = 0;

    // Find the actual length of the expression to avoid processing padding
    int32_t len = 0;
    while (len < M_prefix && prefix_acc[b][len] != NO_OP)
    {
        len++;
    }

    // Iterate from RIGHT to LEFT for prefix notation
    for (int j = len - 1; j >= 0; --j)
    {
        int64_t token_id = prefix_acc[b][j];

        int arity = get_arity(token_id);

        if (arity == 0)
        { // Operand
            if (stack_ptr >= HARD_MAX_LENGTH * 2 || workspace_ptr >= HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = 1;
            workspace[workspace_ptr++] = token_id;
            stack_ptr++;
        }
        else if (arity == 1)
        { // Unary operator
            if (stack_ptr < 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
                break;
            }
            stack_ptr--;
            int32_t op_start = stack_starts[stack_ptr];
            int32_t op_len = stack_lens[stack_ptr];

            // Functional style: op(operand)
            if (is_functional_style(token_id))
            {
                if (workspace_ptr + 3 + op_len > HARD_MAX_LENGTH * 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    break;
                }
                stack_starts[stack_ptr] = workspace_ptr;
                stack_lens[stack_ptr] = 3 + op_len;
                workspace[workspace_ptr++] = token_id;
                workspace[workspace_ptr++] = lparen_id;
                for (int k = 0; k < op_len; ++k)
                    workspace[workspace_ptr++] = workspace[op_start + k];
                workspace[workspace_ptr++] = rparen_id;
            }
            else
            { // Postfix style: (operand)^2
                if (workspace_ptr + 3 + op_len > HARD_MAX_LENGTH * 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    break;
                }
                stack_starts[stack_ptr] = workspace_ptr;
                stack_lens[stack_ptr] = 3 + op_len;
                workspace[workspace_ptr++] = lparen_id;
                for (int k = 0; k < op_len; ++k)
                    workspace[workspace_ptr++] = workspace[op_start + k];
                workspace[workspace_ptr++] = rparen_id;
                workspace[workspace_ptr++] = token_id;
            }
            stack_ptr++;
        }
        else if (arity == 2)
        { // Binary operator
            if (stack_ptr < 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }
            // Note the order of popping for prefix is reversed from postfix
            stack_ptr--;
            int32_t op1_start = stack_starts[stack_ptr];
            int32_t op1_len = stack_lens[stack_ptr];
            stack_ptr--;
            int32_t op2_start = stack_starts[stack_ptr];
            int32_t op2_len = stack_lens[stack_ptr];

            if (workspace_ptr + 3 + op1_len + op2_len > HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = 3 + op1_len + op2_len;
            workspace[workspace_ptr++] = lparen_id;
            for (int k = 0; k < op1_len; ++k)
                workspace[workspace_ptr++] = workspace[op1_start + k];
            workspace[workspace_ptr++] = token_id;
            for (int k = 0; k < op2_len; ++k)
                workspace[workspace_ptr++] = workspace[op2_start + k];
            workspace[workspace_ptr++] = rparen_id;
            stack_ptr++;
        }
    }

    if (errors_acc[b] == 0)
    {
        if (stack_ptr != 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
        }
        else
        {
            int32_t final_len = stack_lens[0];
            if (final_len > M_infix)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
            }
            else
            {
                int32_t final_start = stack_starts[0];
                for (int k = 0; k < final_len; ++k)
                {
                    infix_acc[b][k] = workspace[final_start + k];
                }
            }
        }
    }
}

void prefix_to_infix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_prefix, int64_t M_infix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    prefix_to_infix_kernel<<<blocks, threads>>>(
        prefix_acc, infix_acc, errors_acc, lparen_id, rparen_id, B, M_prefix, M_infix);
}

__global__ void prefix_to_postfix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix, int64_t M_postfix)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = 0;

    // Per-thread workspace for building sub-expressions
    int64_t workspace[HARD_MAX_LENGTH * 2];
    int32_t workspace_ptr = 0;

    // Stack holds pointers (start_idx, len) into the workspace
    int32_t stack_starts[HARD_MAX_LENGTH * 2];
    int32_t stack_lens[HARD_MAX_LENGTH * 2];
    int32_t stack_ptr = 0;

    // Find the actual length of the expression to avoid processing padding
    int32_t len = 0;
    while (len < M_prefix && prefix_acc[b][len] != NO_OP)
    {
        len++;
    }

    // Iterate from RIGHT to LEFT for prefix notation
    for (int j = len - 1; j >= 0; --j)
    {
        int64_t token_id = prefix_acc[b][j];

        int arity = get_arity(token_id);

        if (arity == 0)
        { // Operand (variable or constant)
            if (stack_ptr >= HARD_MAX_LENGTH * 2 || workspace_ptr >= HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = 1;
            workspace[workspace_ptr++] = token_id;
            stack_ptr++;
        }
        else if (arity == 1)
        { // Unary operator
            if (stack_ptr < 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
                break;
            }
            stack_ptr--;
            int32_t operand_start = stack_starts[stack_ptr];
            int32_t operand_len = stack_lens[stack_ptr];

            if (workspace_ptr + operand_len + 1 > HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = operand_len + 1;
            // Copy operand first, then operator (postfix order)
            for (int k = 0; k < operand_len; ++k)
                workspace[workspace_ptr++] = workspace[operand_start + k];
            workspace[workspace_ptr++] = token_id;
            stack_ptr++;
        }
        else if (arity == 2)
        { // Binary operator
            if (stack_ptr < 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }
            // For prefix to postfix, the first operand popped is the left operand
            stack_ptr--;
            int32_t left_start = stack_starts[stack_ptr];
            int32_t left_len = stack_lens[stack_ptr];
            stack_ptr--;
            int32_t right_start = stack_starts[stack_ptr];
            int32_t right_len = stack_lens[stack_ptr];

            if (workspace_ptr + left_len + right_len + 1 > HARD_MAX_LENGTH * 2)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                break;
            }
            stack_starts[stack_ptr] = workspace_ptr;
            stack_lens[stack_ptr] = left_len + right_len + 1;
            // Copy left operand, then right operand, then operator (postfix order)
            for (int k = 0; k < left_len; ++k)
                workspace[workspace_ptr++] = workspace[left_start + k];
            for (int k = 0; k < right_len; ++k)
                workspace[workspace_ptr++] = workspace[right_start + k];
            workspace[workspace_ptr++] = token_id;
            stack_ptr++;
        }
    }

    if (errors_acc[b] == 0)
    {
        if (stack_ptr != 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
        }
        else
        {
            int32_t final_len = stack_lens[0];
            if (final_len > M_postfix)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_POSTFIX_TOO_LONG);
            }
            else
            {
                int32_t final_start = stack_starts[0];
                for (int k = 0; k < final_len; ++k)
                {
                    postfix_acc[b][k] = workspace[final_start + k];
                }
                // Add padding
                for (int k = final_len; k < M_postfix; ++k)
                {
                    postfix_acc[b][k] = NO_OP;
                }
            }
        }
    }
}

void prefix_to_postfix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix, int64_t M_postfix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    prefix_to_postfix_kernel<<<blocks, threads>>>(
        prefix_acc, postfix_acc, errors_acc, B, M_prefix, M_postfix);
}