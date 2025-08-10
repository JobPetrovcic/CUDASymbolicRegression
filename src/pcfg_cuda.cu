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
        errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_MAX_TRIES_EXCEEDED);
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
    bool ended = false;
    for (int32_t i = 0; i < M; ++i)
    {
        if (expressions[b][i] != NO_OP)
        {
            if (ended)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
            expression_len++;
        }
        else
        {
            ended = true;
        }
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

    bool ended = false;
    for (int32_t i = 0; i < M && expressions[b][i] != NO_OP; ++i)
    {
        if (expressions[b][i] != NO_OP)
        {
            if (ended)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
        }
        else
        {
            ended = true;
        }
    }

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
    bool ended = false;
    for (int32_t i = 0; i < M && expressions[b][i] != NO_OP; ++i)
    {
        if (expressions[b][i] != NO_OP)
        {
            if (ended)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
            expression_len++;
        }
        else
        {
            ended = true;
        }
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

    bool ended = false;
    for (int32_t i = 0; i < M && expressions[b][i] != NO_OP; ++i)
    {
        if (expressions[b][i] != NO_OP)
        {
            if (ended)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
        }
        else
        {
            ended = true;
        }
    }

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

    bool started = false;
    int32_t expression_len = 0;
    int32_t balance_check = 0;
    int32_t roots = 0;
    for (int k = M_postfix - 1; k >= 0; --k)
    {
        int64_t token = postfix_acc[b][k];
        if (token == NO_OP)
        {
            if (started)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
        }
        else
        {
            started = true;
            expression_len++;
            balance_check += get_arity(token) - 1;
            if (balance_check == -1)
            {
                roots++;
            }
        }
    }

    // If the expression is empty, we are done. Pad output and return.
    if (expression_len == 0)
    {
        for (int k = 0; k < M_infix; ++k)
        {
            infix_acc[b][k] = NO_OP;
        }
        return;
    }

    if (roots > 1)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS); // More than one root in the expression
        return;
    }

    // --- Start: Right-to-Left Conversion (only runs for valid expressions) ---
    constexpr int32_t AWAITING_RIGHT_CHILD = 0;
    constexpr int32_t AWAITING_LEFT_CHILD = 1;

    int32_t op_stack[HARD_MAX_LENGTH];
    int32_t state_stack[HARD_MAX_LENGTH];
    int32_t stack_ptr = 0;

    int64_t temp_buffer[HARD_MAX_LENGTH * 3];
    int32_t temp_idx = (HARD_MAX_LENGTH * 3) - 1;

    int32_t postfix_idx = expression_len - 1;
    while (postfix_idx >= 0)
    {
        int64_t token = postfix_acc[b][postfix_idx--];
        int arity = get_arity(token);

        if (arity == 0)
        { // Operand
            if (temp_idx < 0)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }
            temp_buffer[temp_idx--] = token;

            bool child_is_done = true;
            while (child_is_done && stack_ptr > 0)
            {
                int32_t parent_op = op_stack[stack_ptr - 1];
                int32_t parent_state = state_stack[stack_ptr - 1];
                if (get_arity(parent_op) == 1)
                { // Unary Parent
                    if (is_functional_style(parent_op))
                    {
                        if (temp_idx - 1 < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            return;
                        }
                        temp_buffer[temp_idx--] = lparen_id;
                        temp_buffer[temp_idx--] = parent_op;
                    }
                    else
                    {
                        if (temp_idx < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            return;
                        }
                        temp_buffer[temp_idx--] = lparen_id;
                    }
                    stack_ptr--;
                }
                else
                { // Binary Parent
                    if (parent_state == AWAITING_RIGHT_CHILD)
                    {
                        if (temp_idx < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            return;
                        }
                        temp_buffer[temp_idx--] = parent_op;
                        state_stack[stack_ptr - 1] = AWAITING_LEFT_CHILD;
                        child_is_done = false;
                    }
                    else
                    { // AWAITING_LEFT_CHILD
                        if (temp_idx < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            return;
                        }
                        temp_buffer[temp_idx--] = lparen_id;
                        stack_ptr--;
                    }
                }
            }
        }
        else
        { // Operator
            if (stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }
            op_stack[stack_ptr] = token;
            state_stack[stack_ptr] = AWAITING_RIGHT_CHILD;

            if (arity == 1 && !is_functional_style(token))
            {
                if (temp_idx < 0)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    return;
                }
                temp_buffer[temp_idx--] = token;
            }
            if (temp_idx < 0)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }
            temp_buffer[temp_idx--] = rparen_id;
            stack_ptr++;
        }
    }

    if (stack_ptr > 0)
    {
        if (get_arity(op_stack[stack_ptr - 1]) == 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
        }
        else
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
        }
    }

    if (errors_acc[b] != 0)
        return;

    // --- Final Copy to Output ---
    int32_t final_len = (HARD_MAX_LENGTH * 3 - 1) - temp_idx;
    int32_t temp_start_idx = temp_idx + 1;

    if (final_len > M_infix)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
    }
    else
    {
        for (int k = 0; k < final_len; ++k)
        {
            infix_acc[b][k] = temp_buffer[temp_start_idx + k];
        }
        for (int k = final_len; k < M_infix; ++k)
        {
            infix_acc[b][k] = NO_OP;
        }
    }
}

// TODO: why does cpu error not raise error?
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

    int32_t expression_len = 0;
    int32_t balance_check = 0;
    int32_t roots = 0;
    bool ended = false;
    for (int32_t i = 0; i < M_prefix; ++i)
    {
        if (prefix_acc[b][i] == NO_OP)
        {
            ended = true;
        }
        else
        {
            if (ended)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
            expression_len++;
            balance_check += get_arity(prefix_acc[b][i]) - 1;
            if (balance_check == -1)
            {
                roots++;
            }
        }
    }

    if (expression_len == 0)
    {
        for (int k = 0; k < M_infix; ++k)
        {
            infix_acc[b][k] = NO_OP;
        }
        return;
    }
    if (roots > 1)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS); // More than one root in the expression
        return;
    }

    constexpr int32_t AWAITING_LEFT_CHILD = 0;
    constexpr int32_t AWAITING_RIGHT_CHILD = 1;

    int32_t op_stack[HARD_MAX_LENGTH];
    int32_t state_stack[HARD_MAX_LENGTH];
    int32_t stack_ptr = 0;

    int32_t prefix_idx = 0;
    int32_t infix_idx = 0;

    while (prefix_idx < expression_len)
    {
        int64_t token = prefix_acc[b][prefix_idx++];
        int arity = get_arity(token);

        if (arity == 0)
        { // Operand
            if (infix_idx >= M_infix)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                return;
            }
            infix_acc[b][infix_idx++] = token;

            bool child_is_done = true;
            while (child_is_done && stack_ptr > 0)
            {
                int32_t parent_op = op_stack[stack_ptr - 1];
                int32_t parent_state = state_stack[stack_ptr - 1];

                if (get_arity(parent_op) == 1)
                { // Unary Parent
                    if (infix_idx >= M_infix)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                        return;
                    }
                    infix_acc[b][infix_idx++] = rparen_id;
                    if (!is_functional_style(parent_op))
                    {
                        if (infix_idx >= M_infix)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                            return;
                        }
                        infix_acc[b][infix_idx++] = parent_op;
                    }
                    stack_ptr--;
                }
                else
                { // Binary Parent
                    if (parent_state == AWAITING_LEFT_CHILD)
                    {
                        if (infix_idx >= M_infix)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                            return;
                        }
                        infix_acc[b][infix_idx++] = parent_op;
                        state_stack[stack_ptr - 1] = AWAITING_RIGHT_CHILD;
                        child_is_done = false;
                    }
                    else
                    { // AWAITING_RIGHT_CHILD
                        if (infix_idx >= M_infix)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                            return;
                        }
                        infix_acc[b][infix_idx++] = rparen_id;
                        stack_ptr--;
                    }
                }
            }
        }
        else
        { // Operator
            if (stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }
            op_stack[stack_ptr] = token;
            state_stack[stack_ptr] = AWAITING_LEFT_CHILD;

            if (arity == 1 && is_functional_style(token))
            {
                if (infix_idx + 2 > M_infix)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                    return;
                }
                infix_acc[b][infix_idx++] = token;
                infix_acc[b][infix_idx++] = lparen_id;
            }
            else
            {
                if (infix_idx + 1 > M_infix)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                    return;
                }
                infix_acc[b][infix_idx++] = lparen_id;
            }
            stack_ptr++;
        }
    }

    if (errors_acc[b] == 0 && stack_ptr != 0)
    {
        if (get_arity(op_stack[stack_ptr - 1]) == 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
        }
        else
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
        }
    }

    for (int k = infix_idx; k < M_infix; ++k)
    {
        infix_acc[b][k] = NO_OP;
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

__global__ void prefix_to_postfix_parent_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> parents_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = 0;

    int32_t expression_len = 0;
    while (expression_len < M && prefix_acc[b][expression_len] != NO_OP)
    {
        expression_len++;
    }
    if (expression_len == 0)
        return;

    // --- Pass 1: Build parent pointers for the prefix tree ---
    int32_t prefix_parents[HARD_MAX_LENGTH];
    int32_t child_stack[HARD_MAX_LENGTH];
    int32_t child_stack_ptr = 0;

    for (int32_t i = 0; i < expression_len; ++i)
    {
        prefix_parents[i] = NULL_PARENT;
    }

    for (int32_t i = expression_len - 1; i >= 0; i--)
    {
        int64_t token = prefix_acc[b][i];
        int arity = get_arity(token);
        if (arity > 0)
        {
            if (child_stack_ptr < arity)
            {
                errors_acc[b] = (arity == 1) ? static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND)
                                             : static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                return;
            }
            for (int j = 0; j < arity; ++j)
            {
                int32_t child_idx = child_stack[--child_stack_ptr];
                prefix_parents[child_idx] = i;
            }
        }
        if (child_stack_ptr >= HARD_MAX_LENGTH)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
            return;
        }
        child_stack[child_stack_ptr++] = i;
    }

    if (child_stack_ptr != 1)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
        return;
    }

    // --- Pass 2: Perform iterative post-order traversal to build postfix and parent arrays ---
    int32_t traversal_stack[HARD_MAX_LENGTH];
    int32_t traversal_stack_ptr = 0;

    int32_t prefix_to_postfix_map[HARD_MAX_LENGTH]; // Map prefix index to its new postfix index
    int32_t postfix_idx = 0;

    traversal_stack[traversal_stack_ptr++] = 0; // Start traversal at the root (index 0)
    int32_t last_node_visited = -1;

    while (traversal_stack_ptr > 0)
    {
        int32_t current_prefix_idx = traversal_stack[traversal_stack_ptr - 1];
        int64_t current_token = prefix_acc[b][current_prefix_idx];
        int arity = get_arity(current_token);

        // Find children of the current node
        int32_t children_indices[MAX_ARITY];
        int32_t child_count = 0;
        if (arity > 0)
        {
            for (int32_t i = current_prefix_idx + 1; i < expression_len && child_count < arity; ++i)
            {
                if (prefix_parents[i] == current_prefix_idx)
                {
                    children_indices[child_count++] = i;
                }
            }
        }

        // If we are coming down the tree and there are unvisited children
        if (child_count > 0 && last_node_visited != children_indices[child_count - 1] && (child_count == 1 || last_node_visited != children_indices[0]))
        {
            for (int i = child_count - 1; i >= 0; --i)
            {
                if (traversal_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    return;
                }
                traversal_stack[traversal_stack_ptr++] = children_indices[i];
            }
        }
        else // We are coming up the tree (children are visited) or it's a leaf
        {
            if (postfix_idx >= M)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_POSTFIX_TOO_LONG);
                return;
            }

            // Write to postfix array
            postfix_acc[b][postfix_idx] = current_token;

            // Map prefix index to its new postfix index
            prefix_to_postfix_map[current_prefix_idx] = postfix_idx;

            // Set the parent pointer in the final parents_acc array
            int32_t parent_prefix_idx = prefix_parents[current_prefix_idx];
            if (parent_prefix_idx != NULL_PARENT)
            {
                parents_acc[b][postfix_idx] = prefix_to_postfix_map[parent_prefix_idx];
            }
            else
            {
                parents_acc[b][postfix_idx] = NULL_PARENT;
            }

            postfix_idx++;
            last_node_visited = traversal_stack[--traversal_stack_ptr];
        }
    }

    if (postfix_idx != expression_len)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
        return;
    }

    // Add padding to the end of the output arrays
    for (int32_t i = postfix_idx; i < M; ++i)
    {
        postfix_acc[b][i] = NO_OP;
        parents_acc[b][i] = NULL_CHILD; // Use NULL_CHILD for padding in parent array for consistency
    }
}

void prefix_to_postfix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> parents_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix, int64_t M_postfix)
{
    // The maximum length of the output is the same as the input.
    TORCH_CHECK(M_prefix == M_postfix, "prefix_to_postfix_parent expects input and output to have the same max length.");

    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    prefix_to_postfix_parent_kernel<<<blocks, threads>>>(
        prefix_acc,
        postfix_acc,
        parents_acc,
        errors_acc,
        B, M_prefix);
}