#include "pcfg_kernels.h"
#include "operators.h"
#include "error_codes.h"
#include <torch/torch.h>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>

void sample_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> rule_lhs,
    const torch::TensorAccessor<int64_t, 1> rhs_ptr,
    const torch::TensorAccessor<int64_t, 1> rhs_concat,
    const torch::TensorAccessor<int64_t, 1> nt_rule_ptr,
    const torch::TensorAccessor<float, 1> nt_rule_cum_probs,
    int64_t start_symbol_id,
    int64_t max_length,
    int64_t max_tries,
    torch::TensorAccessor<int64_t, 2> output,
    torch::TensorAccessor<int64_t, 1> errors,
    const torch::TensorAccessor<int64_t, 1> seeds,
    int64_t B)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        std::mt19937 gen(seeds[b]);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

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
                    float rand_val = dis(gen);
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
                        errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_NO_VALID_RULE_FOUND);
                        break; // Breaks from the while loop
                    }

                    int32_t rhs_start = rhs_ptr[chosen_rule];
                    int32_t rhs_end = rhs_ptr[chosen_rule + 1];
                    for (int32_t i = rhs_end - 1; i >= rhs_start; --i)
                    {
                        if (stack_ptr >= HARD_MAX_LENGTH)
                        {
                            should_restart = true;
                            break;
                        }
                        stack[stack_ptr++] = rhs_concat[i];
                    }
                    if (should_restart)
                    {
                        break; // Breaks from the while loop
                    }
                }
                else
                { // Terminal
                    if (out_ptr >= max_length)
                    {
                        should_restart = true;
                        break; // Breaks from the while loop
                    }
                    output[b][out_ptr++] = current_symbol;
                }
            } // end while

            if (errors[b] != 0)
            {
                break; // An error occurred, break from the for(t) loop
            }
            if (should_restart)
            {
                continue; // Restart the generation for this batch item
            }

            if (stack_ptr == 0 && out_ptr > 0)
            {
                // Successfully generated an expression
                for (int32_t i = out_ptr; i < max_length; ++i)
                {
                    output[b][i] = NO_OP; // padding
                }
                generated_successfully = true;
                errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);
                break; // Success, break from the for(t) loop
            }
        } // end for(t)

        if (!generated_successfully && errors[b] == 0)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_MAX_TRIES_EXCEEDED);
        }
    }
}

void infix_to_prefix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence,
    const torch::TensorAccessor<int64_t, 2> infix_ops,
    torch::TensorAccessor<int64_t, 2> prefix_ops,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        { // Use do-while(false) to allow `break` as a single-level `goto`
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
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN);
                        break;
                    }
                    expression_len++;
                }
                else
                {
                    ended = true;
                }
            }
            if (errors[b] != 0)
                break;

            for (int32_t i = expression_len - 1; i >= 0; --i)
            {
                const int32_t token = infix_ops[b][i];
                if (precedence[token] == 0)
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                        break;
                    }
                    out_queue[out_queue_size++] = token;
                }
                else if (token == rparenthesis_id)
                {
                    if (op_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[op_stack_ptr++] = token;
                }
                else if (token == lparenthesis_id)
                {
                    while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id)
                    {
                        if (out_queue_size >= HARD_MAX_LENGTH)
                        {
                            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                            break;
                        }
                        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                    }
                    if (errors[b] != 0)
                        break;
                    if (op_stack_ptr == 0)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                        break;
                    }
                    op_stack_ptr--;
                }
                else
                {
                    while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id &&
                           ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token]) ||
                            (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token])))
                    {
                        if (out_queue_size >= HARD_MAX_LENGTH)
                        {
                            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                            break;
                        }
                        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                    }
                    if (errors[b] != 0)
                        break;
                    if (op_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[op_stack_ptr++] = token;
                }
            }
            if (errors[b] != 0)
                break;

            while (op_stack_ptr > 0)
            {
                if (op_stack[op_stack_ptr - 1] == rparenthesis_id)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                    break;
                }
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    break;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (errors[b] != 0)
                break;

            for (int32_t i = 0; i < out_queue_size; ++i)
            {
                prefix_ops[b][i] = out_queue[out_queue_size - 1 - i];
            }
            for (int32_t i = out_queue_size; i < M; ++i)
            {
                prefix_ops[b][i] = NO_OP;
            }

        } while (false);
    }
}

void infix_to_postfix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence,
    const torch::TensorAccessor<int64_t, 2> infix_ops,
    torch::TensorAccessor<int64_t, 2> postfix_ops,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        { // Use do-while(false) to allow `break` as a single-level `goto`
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
                else if (ended)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN);
                    break;
                }
            }
            if (errors[b] != 0)
                break;

            for (int32_t i = 0; i < M && infix_ops[b][i] != NO_OP; ++i)
            {
                const int32_t token = infix_ops[b][i];
                if (precedence[token] == 0)
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                        break;
                    }
                    out_queue[out_queue_size++] = token;
                }
                else if (token == lparenthesis_id)
                {
                    if (op_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                        break;
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
                            break;
                        }
                        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                    }
                    if (errors[b] != 0)
                        break;
                    if (op_stack_ptr == 0)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                        break;
                    }
                    op_stack_ptr--;
                }
                else
                {
                    while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                           ((is_left_associative(token) && precedence[op_stack[op_stack_ptr - 1]] >= precedence[token]) ||
                            (is_right_associative(token) && precedence[op_stack[op_stack_ptr - 1]] > precedence[token])))
                    {
                        if (out_queue_size >= HARD_MAX_LENGTH)
                        {
                            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                            break;
                        }
                        out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                    }
                    if (errors[b] != 0)
                        break;
                    if (op_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[op_stack_ptr++] = token;
                }
            }
            if (errors[b] != 0)
                break;

            while (op_stack_ptr > 0)
            {
                if (op_stack[op_stack_ptr - 1] == lparenthesis_id)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                    break;
                }
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                    break;
                }
                out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
            }
            if (errors[b] != 0)
                break; // Exit if shunting-yard itself had an error

            // STAGE 2: VALIDATION (Now happens *before* writing output)
            if (out_queue_size > 0)
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
                    { // It's an operator
                        int32_t arity = get_arity(token);
                        if (validation_stack_size < arity)
                        {
                            if (arity == 1)
                            {
                                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                            }
                            else if (arity == 2)
                            {
                                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                            }
                            else
                            {
                                errors[b] = static_cast<int64_t>(ErrorCode::INTERNAL_ERROR);
                            }
                            break; // Break from the validation for-loop
                        }
                        validation_stack_size = validation_stack_size - arity + 1;
                    }
                }

                if (errors[b] != 0)
                {
                    // An arity error was found and set inside the loop.
                    // Break from the main do-while to prevent further checks and output writing.
                    break;
                }

                if (validation_stack_size != 1)
                {
                    // This now correctly catches errors like "a b" (stack_size=2)
                    // or cases where the inner loop didn't find an arity error but the final state is wrong.
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
                    break;
                }
            }

            // STAGE 3: WRITE OUTPUT (Only if no errors have occurred so far)
            // This block is only reached if errors[b] is still NO_ERROR.
            for (int32_t i = 0; i < out_queue_size; ++i)
            {
                postfix_ops[b][i] = out_queue[i];
            }
            for (int32_t i = out_queue_size; i < M; ++i)
            {
                postfix_ops[b][i] = NO_OP;
            }

        } while (false);
    }
}

void postfix_to_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 2> infix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix)
{
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        {
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
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN);
                        break;
                    }
                }
                else
                {
                    started = true;
                    expression_len++;
                    balance_check += get_arity(token) - 1;
                    if (balance_check == -1)
                        roots++;
                }
            }
            if (errors_acc[b] != 0)
                break;

            if (expression_len == 0)
            {
                for (int32_t k = 0; k < M_infix; ++k)
                    infix_acc[b][k] = NO_OP;
                break; // Exit do-while, effectively a continue
            }

            if (roots > 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
                break;
            }

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
                {
                    if (temp_idx < 0)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    temp_buffer[temp_idx--] = token;

                    bool child_is_done = true;
                    while (child_is_done && stack_ptr > 0)
                    {
                        int32_t parent_op = op_stack[stack_ptr - 1];
                        int32_t parent_state = state_stack[stack_ptr - 1];
                        if (get_arity(parent_op) == 1)
                        {
                            if (is_functional_style(parent_op))
                            {
                                if (temp_idx - 1 < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_buffer[temp_idx--] = lparen_id;
                                temp_buffer[temp_idx--] = parent_op;
                            }
                            else
                            {
                                if (temp_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_buffer[temp_idx--] = lparen_id;
                            }
                            stack_ptr--;
                        }
                        else
                        {
                            if (parent_state == AWAITING_RIGHT_CHILD)
                            {
                                if (temp_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_buffer[temp_idx--] = parent_op;
                                state_stack[stack_ptr - 1] = AWAITING_LEFT_CHILD;
                                child_is_done = false;
                            }
                            else
                            {
                                if (temp_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_buffer[temp_idx--] = lparen_id;
                                stack_ptr--;
                            }
                        }
                    }
                }
                else
                {
                    if (stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[stack_ptr] = token;
                    state_stack[stack_ptr] = AWAITING_RIGHT_CHILD;
                    if (arity == 1 && !is_functional_style(token))
                    {
                        if (temp_idx < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            break;
                        }
                        temp_buffer[temp_idx--] = token;
                    }
                    if (temp_idx < 0)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    temp_buffer[temp_idx--] = rparen_id;
                    stack_ptr++;
                }
                if (errors_acc[b] != 0)
                    break;
            }
            if (errors_acc[b] != 0)
                break;

            if (stack_ptr > 0)
            {
                errors_acc[b] = (get_arity(op_stack[stack_ptr - 1]) == 1)
                                    ? static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND)
                                    : static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }

            int32_t final_len = (HARD_MAX_LENGTH * 3 - 1) - temp_idx;
            if (final_len > M_infix)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
            }
            else
            {
                int32_t temp_start_idx = temp_idx + 1;
                for (int32_t k = 0; k < final_len; ++k)
                    infix_acc[b][k] = temp_buffer[temp_start_idx + k];
                for (int32_t k = final_len; k < M_infix; ++k)
                    infix_acc[b][k] = NO_OP;
            }
        } while (false);
    }
}

void prefix_to_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 2> infix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_prefix, int64_t M_infix)
{
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        {
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
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN);
                        break;
                    }
                    expression_len++;
                    balance_check += get_arity(prefix_acc[b][i]) - 1;
                    if (balance_check == -1)
                        roots++;
                }
            }
            if (errors_acc[b] != 0)
                break;

            if (expression_len == 0)
            {
                for (int32_t k = 0; k < M_infix; ++k)
                    infix_acc[b][k] = NO_OP;
                break;
            }
            if (roots > 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
                break;
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
                {
                    if (infix_idx >= M_infix)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                        break;
                    }
                    infix_acc[b][infix_idx++] = token;

                    bool child_is_done = true;
                    while (child_is_done && stack_ptr > 0)
                    {
                        int32_t parent_op = op_stack[stack_ptr - 1];
                        int32_t parent_state = state_stack[stack_ptr - 1];
                        if (get_arity(parent_op) == 1)
                        {
                            if (infix_idx >= M_infix)
                            {
                                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                                break;
                            }
                            infix_acc[b][infix_idx++] = rparen_id;
                            if (!is_functional_style(parent_op))
                            {
                                if (infix_idx >= M_infix)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                                    break;
                                }
                                infix_acc[b][infix_idx++] = parent_op;
                            }
                            stack_ptr--;
                        }
                        else
                        {
                            if (parent_state == AWAITING_LEFT_CHILD)
                            {
                                if (infix_idx >= M_infix)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                                    break;
                                }
                                infix_acc[b][infix_idx++] = parent_op;
                                state_stack[stack_ptr - 1] = AWAITING_RIGHT_CHILD;
                                child_is_done = false;
                            }
                            else
                            {
                                if (infix_idx >= M_infix)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                                    break;
                                }
                                infix_acc[b][infix_idx++] = rparen_id;
                                stack_ptr--;
                            }
                        }
                    }
                }
                else
                {
                    if (stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[stack_ptr] = token;
                    state_stack[stack_ptr] = AWAITING_LEFT_CHILD;
                    if (arity == 1 && is_functional_style(token))
                    {
                        if (infix_idx + 2 > M_infix)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                            break;
                        }
                        infix_acc[b][infix_idx++] = token;
                        infix_acc[b][infix_idx++] = lparen_id;
                    }
                    else
                    {
                        if (infix_idx + 1 > M_infix)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                            break;
                        }
                        infix_acc[b][infix_idx++] = lparen_id;
                    }
                    stack_ptr++;
                }
                if (errors_acc[b] != 0)
                    break;
            }
            if (errors_acc[b] != 0)
                break;

            if (stack_ptr != 0)
            {
                errors_acc[b] = (get_arity(op_stack[stack_ptr - 1]) == 1)
                                    ? static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND)
                                    : static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }

            for (int32_t k = infix_idx; k < M_infix; ++k)
                infix_acc[b][k] = NO_OP;

        } while (false);
    }
}

void prefix_to_postfix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        {
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
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN);
                        break;
                    }
                    int arity = get_arity(token);
                    balance_check += arity - 1;
                    if (balance_check == -1)
                        roots++;
                    if (parent_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
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
                                break;
                            }
                            postfix_acc[b][output_ptr++] = prefix_acc[b][parent_stack[parent_stack_ptr]];
                            if (parent_stack_ptr > 0)
                                parent_stack_child_count[parent_stack_ptr - 1]--;
                        }
                        else
                            break;
                    }
                }
                if (errors_acc[b] != 0)
                    break;
            }
            if (errors_acc[b] != 0)
                break;

            if (parent_stack_ptr != 0)
            {
                errors_acc[b] = (get_arity(prefix_acc[b][parent_stack[parent_stack_ptr - 1]]) == 1)
                                    ? static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND)
                                    : static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }
            if (roots != 1 && output_ptr > 0)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
                break;
            }

            for (int32_t i = output_ptr; i < M_prefix; ++i)
                postfix_acc[b][i] = NO_OP;

        } while (false);
    }
}

void postfix_to_prefix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t B, int64_t M_postfix)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        {
            int32_t parent_stack[HARD_MAX_LENGTH];
            int32_t parent_stack_child_count[HARD_MAX_LENGTH];
            int32_t parent_stack_ptr = 0;
            int32_t output_ptr = 0;
            std::vector<int64_t> temp_prefix(M_postfix);
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
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNEXPECTED_TOKEN);
                        break;
                    }
                }
                else
                {
                    started = true;
                    int arity = get_arity(token);
                    balance_check += arity - 1;
                    if (balance_check == -1)
                        roots++;
                    if (parent_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    parent_stack[parent_stack_ptr] = i;
                    parent_stack_child_count[parent_stack_ptr++] = arity;
                    while (parent_stack_ptr > 0)
                    {
                        if (parent_stack_child_count[parent_stack_ptr - 1] == 0)
                        {
                            parent_stack_ptr--;
                            if (output_ptr >= M_postfix)
                            {
                                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_PREFIX_TOO_LONG);
                                break;
                            }
                            temp_prefix[output_ptr++] = postfix_acc[b][parent_stack[parent_stack_ptr]];
                            if (parent_stack_ptr > 0)
                                parent_stack_child_count[parent_stack_ptr - 1]--;
                        }
                        else
                            break;
                    }
                }
                if (errors_acc[b] != 0)
                    break;
            }
            if (errors_acc[b] != 0)
                break;

            if (parent_stack_ptr != 0)
            {
                errors_acc[b] = (get_arity(postfix_acc[b][parent_stack[parent_stack_ptr - 1]]) == 1)
                                    ? static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND)
                                    : static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                break;
            }
            if (roots != 1 && output_ptr > 0)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
                break;
            }

            for (int32_t i = 0; i < output_ptr; ++i)
                prefix_acc[b][i] = temp_prefix[output_ptr - 1 - i];
            for (int32_t i = output_ptr; i < M_postfix; ++i)
                prefix_acc[b][i] = NO_OP;

        } while (false);
    }
}

void get_prefix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 2> parent_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        for (int32_t i = 0; i < M_prefix; ++i)
            parent_acc[b][i] = NULL_PARENT;

        do
        {
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
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN);
                        break;
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
                            break;
                        }
                        parent_acc[b][node_stack[--node_stack_ptr]] = i;
                    }
                    else if (is_binary(token))
                    {
                        if (node_stack_ptr < 2)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                            break;
                        }
                        parent_acc[b][node_stack[--node_stack_ptr]] = i; // right child
                        parent_acc[b][node_stack[--node_stack_ptr]] = i; // left child
                    }
                    if (node_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
                        break;
                    }
                    node_stack[node_stack_ptr++] = i;
                }
            }
            if (errors_acc[b] != 0)
                break;

            if (started && node_stack_ptr != 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
            }
        } while (false);
    }
}

void get_postfix_children_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 3> children_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t B, int64_t M_postfix)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors_acc[b] = static_cast<int64_t>(ErrorCode::NO_ERROR);

        do
        {
            int32_t child_stack[HARD_MAX_LENGTH];
            int32_t child_stack_ptr = 0;
            bool ended = false;
            int32_t expression_len = 0;

            for (int32_t i = 0; i < M_postfix; i++)
            {
                if (postfix_acc[b][i] == NO_OP)
                {
                    ended = true;
                }
                else
                {
                    expression_len++;
                    if (ended)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_UNEXPECTED_TOKEN);
                        break;
                    }
                    int32_t token = postfix_acc[b][i];
                    if (is_unary(token))
                    {
                        if (child_stack_ptr < 1)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND);
                            break;
                        }
                        children_acc[b][i][0] = child_stack[--child_stack_ptr];
                        children_acc[b][i][1] = NULL_CHILD;
                    }
                    else if (is_binary(token))
                    {
                        if (child_stack_ptr < 2)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                            break;
                        }
                        int32_t right_child_index = child_stack[--child_stack_ptr];
                        int32_t left_child_index = child_stack[--child_stack_ptr];
                        children_acc[b][i][0] = left_child_index;
                        children_acc[b][i][1] = right_child_index;
                    }
                    else
                    {
                        children_acc[b][i][0] = NULL_CHILD;
                        children_acc[b][i][1] = NULL_CHILD;
                    }
                    if (child_stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
                        break;
                    }
                    child_stack[child_stack_ptr++] = i;
                }
            }
            if (errors_acc[b] != 0)
                break;

            if (expression_len > 0 && child_stack_ptr != 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
            }
        } while (false);
    }
}