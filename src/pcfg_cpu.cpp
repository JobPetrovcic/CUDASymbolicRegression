#include "pcfg_kernels.h"
#include "operators.h"
#include "error_codes.h"
#include <torch/torch.h>
#include <random>
#include <omp.h>

void pcfg_sample_string_expression_cpu_impl(
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
        std::mt19937 gen(seeds[b]);
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        errors[b] = 0;
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
                int32_t rule_start = nt_rule_ptr[current_symbol];
                int32_t rule_end = nt_rule_ptr[current_symbol + 1];

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
                        errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_NO_VALID_RULE_FOUND); // No valid rule found
                        break;                                                                       // Stop processing this batch item
                    }

                    int32_t rhs_start = rhs_ptr[chosen_rule];
                    int32_t rhs_end = rhs_ptr[chosen_rule + 1];
                    for (int32_t i = rhs_end - 1; i >= rhs_start; --i)
                    {
                        if (stack_ptr >= HARD_MAX_LENGTH) // Use HARD_MAX_LENGTH for stack capacity
                        {
                            should_restart = true;
                            break;
                        }
                        stack[stack_ptr++] = rhs_concat[i];
                    }
                    if (should_restart)
                    {
                        // Restart the generation for this batch item
                        break;
                    }
                }
                else
                { // Terminal
                    if (out_ptr >= max_length)
                    {
                        should_restart = true;
                        // This is not an error, the generated expression was too long, try again
                        break; // expression too long
                    }
                    output[b][out_ptr++] = current_symbol;
                }
            }
            if (should_restart)
            {
                // Restart the generation for this batch item
                continue;
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
                break;
            }
        }
        if (!generated_successfully && errors[b] == 0)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::GENERATION_MAX_TRIES_EXCEEDED);
        }
    }
}

void parse_to_prefix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> children_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = 0;
        int32_t op_stack[HARD_MAX_LENGTH];
        int32_t op_stack_ptr = 0;
        int32_t out_queue[HARD_MAX_LENGTH];
        int32_t out_queue_size = 0;

        int32_t expression_len = 0;
        while (expression_len < M && expressions_acc[b][expression_len] != NO_OP)
        {
            expression_len++;
        }

        for (int32_t i = expression_len - 1; i >= 0; --i)
        {
            const int32_t &token = expressions_acc[b][i];
            if (precedence_acc[token] == 0)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    break;
                }
                out_queue[out_queue_size++] = token;
            }
            else if (token == rparenthesis_id) // Treat as lparenthesis
            {
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                    break;
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
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr == 0)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                    break;
                }
                op_stack_ptr--; // Pop rparenthesis
            }
            else
            {
                while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != rparenthesis_id &&
                       ((is_left_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] > precedence_acc[token]) || (is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] >= precedence_acc[token])))
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                    break;
                }
                op_stack[op_stack_ptr++] = token;
            }

            if (errors[b] != 0)
                break;
        }

        if (errors[b] != 0)
            continue;

        while (op_stack_ptr > 0)
        {
            if (op_stack[op_stack_ptr - 1] == rparenthesis_id)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                break;
            }
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                break;
            }
            out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
        }

        if (errors[b] != 0)
            continue;

        // Reverse the output queue to get prefix
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            ops_acc[b][i] = out_queue[out_queue_size - 1 - i];
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            ops_acc[b][i] = NO_OP; // padding
        }

        // Re-use out_queue to store the prefix expression for child calculation
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            out_queue[i] = ops_acc[b][i];
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
                    break;
                }
                int32_t child_index = child_stack[--child_stack_ptr];
                children_acc[b][i][0] = child_index;
                children_acc[b][i][1] = NULL_CHILD;
            }
            else if (is_binary(token))
            {
                if (child_stack_ptr < 2)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS); // Binary operator without enough operands
                    break;
                }
                int32_t left_child_index = child_stack[--child_stack_ptr];
                int32_t right_child_index = child_stack[--child_stack_ptr];
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW); // Child stack overflow
                break;
            }
            child_stack[child_stack_ptr++] = i;
        }
        if (out_queue_size > 0 && child_stack_ptr != 1 && errors[b] == 0)
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression (e.g., too many operands)
    }
}

void parse_to_prefix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 2> parents_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = 0;
        int32_t op_stack[HARD_MAX_LENGTH];
        int32_t op_stack_ptr = 0;
        int32_t out_queue[HARD_MAX_LENGTH];
        int32_t out_queue_size = 0;

        int32_t expression_len = 0;
        while (expression_len < M && expressions_acc[b][expression_len] != NO_OP)
        {
            expression_len++;
        }

        for (int32_t i = expression_len - 1; i >= 0; --i)
        {
            const int32_t &token = expressions_acc[b][i];
            if (precedence_acc[token] == 0)
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
                       ((is_left_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] > precedence_acc[token]) ||
                        (is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] >= precedence_acc[token])))
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                    break;
                }
                op_stack[op_stack_ptr++] = token;
            }
            if (errors[b] != 0)
                break;
        }
        if (errors[b] != 0)
            continue;

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
            continue;

        // Reverse the output queue to get prefix and write to ops_acc tensor
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            ops_acc[b][i] = out_queue[out_queue_size - 1 - i];
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            ops_acc[b][i] = NO_OP;
        }

        // Initialize parents_acc tensor
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            parents_acc[b][i] = NULL_PARENT;
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            parents_acc[b][i] = NO_OP;
        }
        if (out_queue_size == 0)
            continue;

        // Re-use out_queue to store the prefix expression for parent calculation
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            out_queue[i] = ops_acc[b][i];
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
                    break;
                }
                int32_t child_index = node_stack[--node_stack_ptr];
                parents_acc[b][child_index] = i;
            }
            else if (is_binary(token))
            {
                if (node_stack_ptr < 2)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                    break;
                }
                int32_t child1_index = node_stack[--node_stack_ptr];
                int32_t child2_index = node_stack[--node_stack_ptr];
                parents_acc[b][child1_index] = i;
                parents_acc[b][child2_index] = i;
            }
            if (node_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
                break;
            }
            node_stack[node_stack_ptr++] = i;
        }
        if (errors[b] != 0)
            continue;

        if (out_queue_size > 0 && node_stack_ptr != 1)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION);
        }
    }
}

void parse_to_postfix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> children_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = 0;
        int32_t op_stack[HARD_MAX_LENGTH];
        int32_t op_stack_ptr = 0;
        int32_t out_queue[HARD_MAX_LENGTH];
        int32_t out_queue_size = 0;

        for (int32_t i = 0; i < M && expressions_acc[b][i] != NO_OP; ++i)
        {
            const int32_t &token = expressions_acc[b][i];
            if (precedence_acc[token] == 0)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                    break;
                }
                out_queue[out_queue_size++] = token;
            }
            else if (token == lparenthesis_id)
            {
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
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
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr == 0)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                    break;
                }
                op_stack_ptr--; // Pop lparenthesis
            }
            else
            {
                while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                       ((is_left_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] >= precedence_acc[token]) ||
                        (is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] > precedence_acc[token])))
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW); // Operator stack overflow
                    break;
                }
                op_stack[op_stack_ptr++] = token;
            }

            if (errors[b] != 0)
                break;
        }

        if (errors[b] != 0)
            continue;

        while (op_stack_ptr > 0)
        {
            if (op_stack[op_stack_ptr - 1] == lparenthesis_id)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS); // Mismatched parenthesis
                break;
            }
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW); // Postfix expression too long
                break;
            }
            out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
        }

        if (errors[b] != 0)
            continue;

        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            ops_acc[b][i] = out_queue[i];
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            ops_acc[b][i] = NO_OP; // padding
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
                    break;
                }
                int32_t child_index = child_stack[--child_stack_ptr];
                children_acc[b][i][0] = child_index;
                children_acc[b][i][1] = NULL_CHILD;
            }
            else if (is_binary(token))
            {
                if (child_stack_ptr < 2)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS); // Binary operator without enough operands
                    break;
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
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW); // Child stack overflow
                break;
            }
            child_stack[child_stack_ptr++] = i;
        }
        if (out_queue_size > 0 && child_stack_ptr != 1 && errors[b] == 0)
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression (e.g., too many operands)
    }
}

void parse_to_postfix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 2> parents_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M)
{
#pragma omp parallel for
    for (int32_t b = 0; b < B; ++b)
    {
        errors[b] = 0;
        int32_t op_stack[HARD_MAX_LENGTH];
        int32_t op_stack_ptr = 0;
        int32_t out_queue[HARD_MAX_LENGTH];
        int32_t out_queue_size = 0;

        for (int32_t i = 0; i < M && expressions_acc[b][i] != NO_OP; ++i)
        {
            const int32_t &token = expressions_acc[b][i];
            if (precedence_acc[token] == 0)
            { // Terminal
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
                if (op_stack_ptr == 0)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_MISMATCHED_PARENTHESIS);
                    break;
                }
                op_stack_ptr--; // Pop lparenthesis
            }
            else
            { // Operator
                while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                       ((is_left_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] >= precedence_acc[token]) ||
                        (is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] > precedence_acc[token])))
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW);
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW);
                    break;
                }
                op_stack[op_stack_ptr++] = token;
            }
            if (errors[b] != 0)
                break;
        }
        if (errors[b] != 0)
            continue;

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
            continue;

        // Write postfix expression to ops_acc tensor
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            ops_acc[b][i] = out_queue[i];
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            ops_acc[b][i] = NO_OP; // padding
        }

        // Initialize parents_acc tensor
        for (int32_t i = 0; i < out_queue_size; ++i)
        {
            parents_acc[b][i] = NULL_PARENT;
        }
        for (int32_t i = out_queue_size; i < M; ++i)
        {
            parents_acc[b][i] = NO_OP; // padding
        }
        if (out_queue_size == 0)
            continue;

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
                    break;
                }
                int32_t child_index = node_stack[--node_stack_ptr];
                parents_acc[b][child_index] = i;
            }
            else if (is_binary(token))
            {
                if (node_stack_ptr < 2)
                {
                    errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS);
                    break;
                }
                int32_t right_child_index = node_stack[--node_stack_ptr];
                int32_t left_child_index = node_stack[--node_stack_ptr];
                parents_acc[b][right_child_index] = i;
                parents_acc[b][left_child_index] = i;
            }
            if (node_stack_ptr >= HARD_MAX_LENGTH)
            {
                errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW);
                break;
            }
            node_stack[node_stack_ptr++] = i;
        }
        if (errors[b] != 0)
            continue;

        if (out_queue_size > 0 && node_stack_ptr != 1)
        {
            errors[b] = static_cast<int64_t>(ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION); // Malformed expression
        }
    }
}

void postfix_to_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 2> infix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    const std::unordered_map<int64_t, std::string> &id_to_symbol,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix)
{
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        std::stack<std::vector<int64_t>> s;
        errors_acc[b] = 0;

        for (int j = 0; j < M_postfix; ++j)
        {
            int64_t token_id = postfix_acc[b][j];
            if (token_id == NO_OP)
                break;

            int arity = get_arity(token_id);
            if (arity == 0)
            { // Operand
                s.push({token_id});
            }
            else if (arity == 1)
            { // Unary operator
                if (s.empty())
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
                    break;
                }
                auto operand = s.top();
                s.pop();

                std::vector<int64_t> new_expr;
                if (is_functional_style(token_id))
                {
                    new_expr.push_back(token_id);
                    new_expr.push_back(lparen_id);
                    new_expr.insert(new_expr.end(), operand.begin(), operand.end());
                    new_expr.push_back(rparen_id);
                }
                else
                {
                    new_expr.push_back(lparen_id);
                    new_expr.insert(new_expr.end(), operand.begin(), operand.end());
                    new_expr.push_back(rparen_id);
                    new_expr.push_back(token_id);
                }
                s.push(new_expr);
            }
            else if (arity == 2)
            { // Binary operator
                if (s.size() < 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                    break;
                }
                auto op2 = s.top();
                s.pop();
                auto op1 = s.top();
                s.pop();

                std::vector<int64_t> new_expr;
                new_expr.push_back(lparen_id);
                new_expr.insert(new_expr.end(), op1.begin(), op1.end());
                new_expr.push_back(token_id);
                new_expr.insert(new_expr.end(), op2.begin(), op2.end());
                new_expr.push_back(rparen_id);
                s.push(new_expr);
            }
        }

        if (errors_acc[b] == 0)
        {
            if (s.size() != 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION); // Malformed expression
            }
            else
            {
                auto final_expr = s.top();
                if (final_expr.size() > M_infix)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG); // Infix expression too long
                }
                else
                {
                    for (size_t k = 0; k < final_expr.size(); ++k)
                    {
                        infix_acc[b][k] = final_expr[k];
                    }
                }
            }
        }
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
        std::stack<std::vector<int64_t>> s;
        errors_acc[b] = 0;

        int64_t len = 0;
        while (len < M_prefix && prefix_acc[b][len] != NO_OP)
        {
            len++;
        }

        for (int j = len - 1; j >= 0; --j)
        {
            int64_t token_id = prefix_acc[b][j];

            int arity = get_arity(token_id);
            if (arity == 0)
            { // Operand
                s.push({token_id});
            }
            else if (arity == 1)
            { // Unary operator
                if (s.empty())
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND);
                    break;
                }
                auto operand = s.top();
                s.pop();

                std::vector<int64_t> new_expr;
                if (is_functional_style(token_id))
                {
                    new_expr.push_back(token_id);
                    new_expr.push_back(lparen_id);
                    new_expr.insert(new_expr.end(), operand.begin(), operand.end());
                    new_expr.push_back(rparen_id);
                }
                else
                { // Postfix style
                    new_expr.push_back(lparen_id);
                    new_expr.insert(new_expr.end(), operand.begin(), operand.end());
                    new_expr.push_back(rparen_id);
                    new_expr.push_back(token_id);
                }
                s.push(new_expr);
            }
            else if (arity == 2)
            { // Binary operator
                if (s.size() < 2)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS);
                    break;
                }
                auto op1 = s.top();
                s.pop();
                auto op2 = s.top();
                s.pop();

                std::vector<int64_t> new_expr;
                new_expr.push_back(lparen_id);
                new_expr.insert(new_expr.end(), op1.begin(), op1.end());
                new_expr.push_back(token_id);
                new_expr.insert(new_expr.end(), op2.begin(), op2.end());
                new_expr.push_back(rparen_id);
                s.push(new_expr);
            }
        }

        if (errors_acc[b] == 0)
        {
            if (s.size() != 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION); // Malformed expression
            }
            else
            {
                auto final_expr = s.top();
                if (final_expr.size() > M_infix)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG); // Infix expression too long
                }
                else
                {
                    for (size_t k = 0; k < final_expr.size(); ++k)
                    {
                        infix_acc[b][k] = final_expr[k];
                    }
                }
            }
        }
    }
}