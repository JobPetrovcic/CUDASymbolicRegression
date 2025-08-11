#include "pcfg_kernels.h"
#include "operators.h"
#include "error_codes.h"
#include <torch/torch.h>
#include <cuda.h>
#include <curand_kernel.h>

constexpr int THREAD_COUNT = 256;

__global__ void sample_infix_kernel(
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

    errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

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

void sample_infix_cuda_impl(
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
    const int blocks = (B + THREAD_COUNT - 1) / THREAD_COUNT;
    sample_infix_kernel<<<blocks, THREAD_COUNT>>>(
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

__global__ void infix_to_prefix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> infix_ops,
    torch::PackedTensorAccessor32<int64_t, 2> prefix_ops,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);
    int32_t op_stack[HARD_MAX_LENGTH];
    int32_t op_stack_ptr = 0;
    int32_t out_queue[HARD_MAX_LENGTH];
    int32_t out_queue_size = 0;

    int32_t expression_len = 0;
    bool ended = false;
    for (int32_t i = 0; i < M; ++i)
    {
        if (infix_ops[b][i] != NO_OP)
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
        const int32_t token = infix_ops[b][i];
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
        prefix_ops[b][i] = out_queue[out_queue_size - 1 - i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        prefix_ops[b][i] = NO_OP; // padding
    }
}

void infix_to_prefix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> infix_ops,
    torch::PackedTensorAccessor32<int64_t, 2> prefix_ops,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    infix_to_prefix_kernel<<<blocks, threads>>>(
        precedence,
        infix_ops,
        prefix_ops,
        errors,
        lparenthesis_id,
        rparenthesis_id,
        B,
        M);
}

__global__ void infix_to_postfix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> infix_ops,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_ops,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);
    int32_t op_stack[HARD_MAX_LENGTH];
    int32_t op_stack_ptr = 0;
    int32_t out_queue[HARD_MAX_LENGTH];
    int32_t out_queue_size = 0;

    bool ended = false;
    for (int32_t i = 0; i < M; ++i)
    {
        if (infix_ops[b][i] == NO_OP)
        {
            ended = true;
        }
        else
        {
            if (ended)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
        }
    }

    for (int32_t i = 0; i < M && infix_ops[b][i] != NO_OP; ++i)
    {
        const int32_t token = infix_ops[b][i];
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

    if (errors[b] != static_cast<int64_t>(ErrorCode::NO_ERROR))
    {
        // An error already occurred during conversion (e.g., overflow), so skip validation.
        return;
    }

    if (out_queue_size == 0)
    {
        // Empty expression is valid, do nothing or handle as a special case if needed.
    }
    else
    {
        int32_t validation_stack_size = 0;
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            const int32_t token = out_queue[i];

            if (precedence[token] == 0)
            { // It's an operand
                validation_stack_size++;
            }
            else
            {                                     // It's an operator
                int32_t arity = get_arity(token); // You need this helper function
                if (validation_stack_size < arity)
                {
                    if (arity == 1)
                    {
                        // Not enough operands for this operator
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                        return;
                    }
                    else if (arity == 2)
                    {
                        // Not enough operands for this operator
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                        return;
                    }
                    else
                    {
                        // Unsupported arity
                        errors[b] = static_cast<int64_t>(ErrorCode::INTERNAL_ERROR);
                        return;
                    }
                }
                validation_stack_size = validation_stack_size - arity + 1;
            }
        }

        if (validation_stack_size != 1)
        {
            // If the final stack size isn't 1, the expression was malformed.
            // This catches cases like "a b" (too many operands) or just "+" (too few).
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
            return;
        }
    }

    for (int32_t i = 0; i < out_queue_size; ++i)
    {
        postfix_ops[b][i] = out_queue[i];
    }
    for (int32_t i = out_queue_size; i < M; ++i)
    {
        postfix_ops[b][i] = NO_OP; // padding
    }
}

void infix_to_postfix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> infix_ops,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_ops,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    infix_to_postfix_kernel<<<blocks, threads>>>(
        precedence,
        infix_ops,
        postfix_ops,
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

    errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

    bool started = false;
    int32_t expression_len = 0;
    int32_t balance_check = 0;
    int32_t roots = 0;
    for (int32_t k = M_postfix - 1; k >= 0; --k)
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
        for (int32_t k = 0; k < M_infix; ++k)
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

    // --- Start: Right-to-Left Conversion (only runs for valid infix_ops) ---
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
        for (int32_t k = 0; k < final_len; ++k)
        {
            infix_acc[b][k] = temp_buffer[temp_start_idx + k];
        }
        for (int32_t k = final_len; k < M_infix; ++k)
        {
            infix_acc[b][k] = NO_OP;
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

    errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

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
        for (int32_t k = 0; k < M_infix; ++k)
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

    for (int32_t k = infix_idx; k < M_infix; ++k)
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

__global__ void prefix_to_postfix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

    // --- Pass 1: Build parent pointers for the prefix tree ---
    int32_t parent_stack[HARD_MAX_LENGTH];
    int32_t parent_stack_child_count[HARD_MAX_LENGTH];
    int32_t parent_stack_ptr = 0;
    int32_t output_ptr = 0;

    bool ended = false;
    int32_t balance_check = 0;
    int32_t roots = 0;
    for (int32_t i = 0; i < M_prefix; ++i)
    {
        int64_t token = prefix_acc[b][i];
        if (token == NO_OP)
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

            int arity = get_arity(token);
            balance_check += arity - 1;
            if (balance_check == -1)
            {
                roots++;
            }

            if (parent_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }

            parent_stack[parent_stack_ptr] = i;
            parent_stack_child_count[parent_stack_ptr++] = arity;

            while (parent_stack_ptr > 0)
            {
                if (parent_stack_child_count[parent_stack_ptr - 1] == 0)
                {
                    parent_stack_ptr--;
                    if (output_ptr >= M_prefix)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_POSTFIX_TOO_LONG);
                        return;
                    }
                    int32_t token_index = parent_stack[parent_stack_ptr];
                    postfix_acc[b][output_ptr++] = prefix_acc[b][token_index];

                    if (parent_stack_ptr > 0)
                    {
                        parent_stack_child_count[parent_stack_ptr - 1]--;
                    }
                }
                else
                {
                    break; // Found a parent that still has children
                }
            }
        }
    }
    if (parent_stack_ptr != 0 && errors_acc[b] == 0)
    {
        if (get_arity(parent_stack[parent_stack_ptr - 1]) == 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
        }
        else
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
        }
    }
    if (roots != 1)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS); // More than one root in the expression
        return;
    }
    for (int32_t i = output_ptr; i < M_prefix; ++i)
    {
        postfix_acc[b][i] = NO_OP; // padding
    }
}

void prefix_to_postfix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    prefix_to_postfix_kernel<<<blocks, threads>>>(
        prefix_acc,
        postfix_acc,
        errors_acc,
        B, M_prefix);
}

__global__ void postfix_to_prefix_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_postfix)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

    // --- Pass 1: Build parent pointers for the prefix tree ---
    int32_t parent_stack[HARD_MAX_LENGTH];
    int32_t parent_stack_child_count[HARD_MAX_LENGTH];
    int32_t parent_stack_ptr = 0;
    int32_t output_ptr = 0;

    bool started = false;
    int32_t balance_check = 0;
    int32_t roots = 0;
    for (int32_t i = M_postfix - 1; i >= 0; --i)
    {
        int64_t token = postfix_acc[b][i];
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

            int arity = get_arity(token);
            balance_check += arity - 1;
            if (balance_check == -1)
            {
                roots++;
            }

            if (parent_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                return;
            }

            parent_stack[parent_stack_ptr] = i;
            parent_stack_child_count[parent_stack_ptr++] = arity;

            while (parent_stack_ptr > 0)
            {
                if (parent_stack_child_count[parent_stack_ptr - 1] == 0)
                {
                    // Pop the parent stack if it has no more children and write it to the output
                    parent_stack_ptr--;
                    if (output_ptr >= M_postfix)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_PREFIX_TOO_LONG);
                        return;
                    }
                    int32_t token_index = parent_stack[parent_stack_ptr];
                    prefix_acc[b][output_ptr++] = postfix_acc[b][token_index];

                    if (parent_stack_ptr > 0)
                    {
                        parent_stack_child_count[parent_stack_ptr - 1]--;
                    }
                }
                else
                {
                    break; // Found a parent that still has children
                }
            }
        }
    }
    if (parent_stack_ptr != 0 && errors_acc[b] == 0)
    {
        if (get_arity(parent_stack[parent_stack_ptr - 1]) == 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
        }
        else
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
        }
    }
    if (roots != 1)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS); // More than one root in the expression
        return;
    }
    // --- Pass 2: Swap to get the correct order for postfix ---
    for (int32_t i = 0; i < output_ptr / 2; ++i)
    {
        int32_t temp = prefix_acc[b][i];
        prefix_acc[b][i] = prefix_acc[b][output_ptr - 1 - i];
        prefix_acc[b][output_ptr - 1 - i] = temp;
    }

    for (int32_t i = output_ptr; i < M_postfix; ++i)
    {
        prefix_acc[b][i] = NO_OP; // padding
    }
}

void postfix_to_prefix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    postfix_to_prefix_kernel<<<blocks, threads>>>(
        postfix_acc,
        prefix_acc,
        errors_acc,
        B, M_prefix);
}

__global__ void get_prefix_parent_cuda_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> parent_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    for (int32_t i = 0; i < M_prefix; ++i)
    {
        parent_acc[b][i] = NULL_PARENT;
    }

    int32_t node_stack[HARD_MAX_LENGTH];
    int32_t node_stack_ptr = 0;
    bool started = false;
    for (int32_t i = M_prefix - 1; i >= 0; i--)
    {
        int32_t token = prefix_acc[b][i];
        if (token == NO_OP)
        {
            if (started)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }
        }
        else
        {
            started = true;

            if (is_unary(token))
            {
                if (node_stack_ptr < 1)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                    return;
                }
                int32_t child_index = node_stack[--node_stack_ptr];
                parent_acc[b][child_index] = i;
            }
            else if (is_binary(token))
            {
                if (node_stack_ptr < 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                    return;
                }
                int32_t right_child_index = node_stack[--node_stack_ptr];
                int32_t left_child_index = node_stack[--node_stack_ptr];
                parent_acc[b][right_child_index] = i;
                parent_acc[b][left_child_index] = i;
            }

            if (node_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
                return;
            }
            node_stack[node_stack_ptr++] = i;
        }
    }
    if (node_stack_ptr != 1 && errors_acc[b] == 0)
        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression
}

void get_prefix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> parent_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    get_prefix_parent_cuda_kernel<<<blocks, threads>>>(
        prefix_acc,
        parent_acc,
        errors_acc,
        B, M_prefix);
}

__global__ void get_postfix_children_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 3> children_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_postfix)
{
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;

    int32_t child_stack[HARD_MAX_LENGTH];
    int32_t child_stack_ptr = 0;
    bool ended = false;
    for (int32_t i = 0; i < M_postfix; i++)
    {
        if (postfix_acc[b][i] == NO_OP)
        {

            ended = true;
        }
        else
        {
            if (ended)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN); // Unexpected token after end
                return;
            }

            int32_t token = postfix_acc[b][i];
            if (is_unary(token))
            {
                if (child_stack_ptr < 1)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND); // Unary operator without operand
                    return;
                }
                int32_t child_index = child_stack[--child_stack_ptr];
                children_acc[b][i][0] = child_index;
                children_acc[b][i][1] = NULL_CHILD;
            }
            else if (is_binary(token))
            {
                if (child_stack_ptr < 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS); // Binary operator without enough operands
                    return;
                }
                int32_t right_child_index = child_stack[--child_stack_ptr];
                int32_t left_child_index = child_stack[--child_stack_ptr];
                children_acc[b][i][0] = left_child_index;
                children_acc[b][i][1] = right_child_index;
            }
            else
            {
                // no children
                children_acc[b][i][0] = NULL_CHILD;
                children_acc[b][i][1] = NULL_CHILD;
            }
            if (child_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW); // Child stack overflow
                return;
            }
            child_stack[child_stack_ptr++] = i;
        }
    }
    if (child_stack_ptr != 1 && errors_acc[b] == 0)
        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression (e.g., too many operands)
}

void get_postfix_children_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 3> children_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t B, int64_t M_postfix)
{
    const int threads = THREAD_COUNT;
    const int blocks = (B + threads - 1) / threads;
    get_postfix_children_kernel<<<blocks, threads>>>(
        postfix_acc,
        children_acc,
        errors_acc,
        B, M_postfix);
}