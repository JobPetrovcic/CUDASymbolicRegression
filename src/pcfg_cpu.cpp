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
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix)
{
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        errors_acc[b] = 0;

        // --- Precheck for multiple roots (mirroring CUDA implementation) ---
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
                {
                    roots++;
                }
            }
        }

        // If the expression is empty, pad output and continue
        if (expression_len == 0)
        {
            for (int k = 0; k < M_infix; ++k)
            {
                infix_acc[b][k] = NO_OP;
            }
            continue;
        }

        if (roots > 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
            continue;
        }

        if (errors_acc[b] != 0)
            continue;

        constexpr int32_t AWAITING_RIGHT_CHILD = 0;
        constexpr int32_t AWAITING_LEFT_CHILD = 1;

        int32_t op_stack[HARD_MAX_LENGTH];
        int32_t state_stack[HARD_MAX_LENGTH];
        int32_t stack_ptr = 0;

        int64_t temp_infix_buffer[HARD_MAX_LENGTH * 3]; // Increased buffer size for safety
        int32_t temp_infix_idx = (HARD_MAX_LENGTH * 3) - 1;

        if (expression_len > 0)
        {
            int32_t postfix_idx = expression_len - 1;
            while (postfix_idx >= 0)
            {
                int64_t token = postfix_acc[b][postfix_idx--];
                int arity = get_arity(token);

                if (arity == 0)
                { // Operand
                    if (temp_infix_idx < 0)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    temp_infix_buffer[temp_infix_idx--] = token;

                    bool child_is_done = true;
                    while (child_is_done && stack_ptr > 0)
                    {
                        int32_t parent_op = op_stack[stack_ptr - 1];
                        int32_t parent_state = state_stack[stack_ptr - 1];

                        if (get_arity(parent_op) == 1)
                        { // Unary Parent
                            if (is_functional_style(parent_op))
                            {
                                if (temp_infix_idx - 1 < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_infix_buffer[temp_infix_idx--] = lparen_id;
                                temp_infix_buffer[temp_infix_idx--] = parent_op;
                            }
                            else
                            {
                                if (temp_infix_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_infix_buffer[temp_infix_idx--] = lparen_id;
                            }
                            stack_ptr--;
                        }
                        else
                        { // Binary Parent
                            if (parent_state == AWAITING_RIGHT_CHILD)
                            {
                                if (temp_infix_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_infix_buffer[temp_infix_idx--] = parent_op;
                                state_stack[stack_ptr - 1] = AWAITING_LEFT_CHILD;
                                child_is_done = false;
                            }
                            else
                            { // AWAITING_LEFT_CHILD
                                if (temp_infix_idx < 0)
                                {
                                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                                    break;
                                }
                                temp_infix_buffer[temp_infix_idx--] = lparen_id;
                                stack_ptr--;
                            }
                        }
                    }
                    if (errors_acc[b] != 0)
                        break;
                }
                else
                { // Operator
                    if (stack_ptr >= HARD_MAX_LENGTH)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    op_stack[stack_ptr] = token;
                    state_stack[stack_ptr] = AWAITING_RIGHT_CHILD;

                    if (arity == 1 && !is_functional_style(token))
                    {
                        if (temp_infix_idx < 0)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            break;
                        }
                        temp_infix_buffer[temp_infix_idx--] = token;
                    }
                    if (temp_infix_idx < 0)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                        break;
                    }
                    temp_infix_buffer[temp_infix_idx--] = rparen_id;
                    stack_ptr++;
                }
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

        if (errors_acc[b] != 0)
            continue;

        int32_t final_len = (HARD_MAX_LENGTH * 3 - 1) - temp_infix_idx;
        int32_t start_idx_in_temp = temp_infix_idx + 1;

        if (final_len > M_infix)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
        }
        else
        {
            for (int32_t k = 0; k < final_len; ++k)
            {
                infix_acc[b][k] = temp_infix_buffer[start_idx_in_temp + k];
            }
            for (int32_t k = final_len; k < M_infix; ++k)
            {
                infix_acc[b][k] = NO_OP;
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
        errors_acc[b] = 0;

        // --- Precheck for multiple roots (mirroring CUDA implementation) ---
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
            continue;
        }
        if (roots > 1)
        {
            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MULTIPLE_ROOTS);
            continue;
        }

        if (errors_acc[b] != 0)
            continue;

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
                    break;
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
                    { // Binary Parent
                        if (parent_state == AWAITING_LEFT_CHILD)
                        {
                            if (infix_idx >= M_infix)
                            {
                                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG);
                                break;
                            }
                            infix_acc[b][infix_idx++] = parent_op;
                            state_stack[stack_ptr - 1] = AWAITING_RIGHT_CHILD;
                            child_is_done = false; // Stop ascending, need right child.
                        }
                        else
                        { // AWAITING_RIGHT_CHILD
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
                if (errors_acc[b] != 0)
                    break;
            }
            else
            { // Operator
                if (stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                    break;
                }
                op_stack[stack_ptr] = token;
                state_stack[stack_ptr] = AWAITING_LEFT_CHILD; // Universal initial state

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
}

void prefix_to_postfix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 2> parents_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t B, int64_t M_prefix, int64_t M_postfix)
{
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        errors_acc[b] = 0;
        bool has_error = false;

        int32_t expression_len = 0;
        while (expression_len < M_prefix && prefix_acc[b][expression_len] != NO_OP)
        {
            expression_len++;
        }

        if (expression_len == 0)
        {
            continue; // Skip empty expressions.
        }

        // --- Pass 1: Build parent pointers for the prefix tree (O(M) time, O(M) space) ---
        int32_t prefix_parents[HARD_MAX_LENGTH];
        { // Scope for Pass 1 variables
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
                        has_error = true;
                        break;
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
                    has_error = true;
                    break;
                }
                child_stack[child_stack_ptr++] = i;
            }

            if (!has_error && child_stack_ptr != 1)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
                has_error = true;
            }
        }

        // --- Pass 2: Iterative post-order traversal to build output (O(M) time, O(M) space) ---
        if (!has_error)
        {
            int32_t traversal_stack[HARD_MAX_LENGTH];
            int32_t traversal_stack_ptr = 0;

            int32_t prefix_to_postfix_map[HARD_MAX_LENGTH];
            int32_t postfix_idx = 0;

            traversal_stack[traversal_stack_ptr++] = 0; // Start traversal at the root
            int32_t last_node_visited = -1;

            while (traversal_stack_ptr > 0)
            {
                int32_t current_prefix_idx = traversal_stack[traversal_stack_ptr - 1];
                int64_t current_token = prefix_acc[b][current_prefix_idx];
                int arity = get_arity(current_token);

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

                if (child_count > 0 && last_node_visited != children_indices[child_count - 1] && (child_count == 1 || last_node_visited != children_indices[0]))
                {
                    for (int i = child_count - 1; i >= 0; --i)
                    {
                        if (traversal_stack_ptr >= HARD_MAX_LENGTH)
                        {
                            errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW);
                            has_error = true;
                            break;
                        }
                        traversal_stack[traversal_stack_ptr++] = children_indices[i];
                    }
                }
                else
                {
                    if (postfix_idx >= M_postfix)
                    {
                        errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_RESULTING_POSTFIX_TOO_LONG);
                        has_error = true;
                        break;
                    }

                    postfix_acc[b][postfix_idx] = current_token;
                    prefix_to_postfix_map[current_prefix_idx] = postfix_idx;

                    int32_t parent_prefix_idx = prefix_parents[current_prefix_idx];
                    parents_acc[b][postfix_idx] = (parent_prefix_idx != NULL_PARENT) ? prefix_to_postfix_map[parent_prefix_idx] : NULL_PARENT;

                    postfix_idx++;
                    last_node_visited = traversal_stack[--traversal_stack_ptr];
                }
                if (has_error)
                    break;
            }

            if (!has_error && postfix_idx != expression_len)
            {
                errors_acc[b] = static_cast<int64_t>(ErrorCode::CONVERSION_MALFORMED_EXPRESSION);
                has_error = true;
            }

            if (!has_error)
            {
                // Add padding to the end of the output arrays
                for (int32_t i = postfix_idx; i < M_postfix; ++i)
                {
                    postfix_acc[b][i] = NO_OP;
                    parents_acc[b][i] = NULL_CHILD;
                }
            }
        }
    }
}
