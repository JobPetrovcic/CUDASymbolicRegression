#include "pcfg_kernels.h"
#include "operators.h"
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
    for (int64_t b = 0; b < B; ++b)
    {
        std::mt19937 gen(seeds[b]);
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        errors[b] = 0;
        bool generated_successfully = false;
        for (int64_t t = 0; t < max_tries; ++t)
        {
            bool should_restart = false;

            int64_t stack[HARD_MAX_LENGTH];
            int64_t stack_ptr = 0;
            stack[stack_ptr++] = start_symbol_id;

            int64_t out_ptr = 0;

            while (stack_ptr > 0)
            {
                int64_t current_symbol = stack[--stack_ptr];
                int64_t rule_start = nt_rule_ptr[current_symbol];
                int64_t rule_end = nt_rule_ptr[current_symbol + 1];

                if (rule_start != rule_end)
                { // Non-terminal
                    float rand_val = dis(gen);
                    int64_t chosen_rule = INVALID_RULE;
                    for (int64_t r = rule_start; r < rule_end; ++r)
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
                        break;          // Stop processing this batch item
                    }

                    int64_t rhs_start = rhs_ptr[chosen_rule];
                    int64_t rhs_end = rhs_ptr[chosen_rule + 1];
                    for (int64_t i = rhs_end - 1; i >= rhs_start; --i)
                    {
                        if (stack_ptr >= max_length)
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
                for (int64_t i = out_ptr; i < max_length; ++i)
                {
                    output[b][i] = NO_OP; // padding
                }
                generated_successfully = true;
                break;
            }
        }
        if (!generated_successfully)
        {
            errors[b] = 9;
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
    for (int b = 0; b < B; ++b)
    {
        errors[b] = 0;
        int64_t op_stack[HARD_MAX_LENGTH];
        int64_t op_stack_ptr = 0;
        int64_t out_queue[HARD_MAX_LENGTH];
        int64_t out_queue_size = 0;

        for (int i = 0; i < M && expressions_acc[b][i] != NO_OP; ++i)
        {
            const int64_t &token = expressions_acc[b][i];
            if (precedence_acc[token] == 0)
            {
                if (out_queue_size >= HARD_MAX_LENGTH)
                {
                    errors[b] = 1; // Postfix expression too long
                    break;
                }
                out_queue[out_queue_size++] = token;
            }
            else if (token == lparenthesis_id)
            {
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = 2; // Operator stack overflow
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
                        errors[b] = 1; // Postfix expression too long
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr == 0)
                {
                    errors[b] = 3; // Mismatched parenthesis
                    break;
                }
                op_stack_ptr--; // Pop lparenthesis
            }
            else
            {
                while (op_stack_ptr > 0 && op_stack[op_stack_ptr - 1] != lparenthesis_id &&
                       ((!is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] >= precedence_acc[token]) ||
                        (is_right_associative(token) && precedence_acc[op_stack[op_stack_ptr - 1]] > precedence_acc[token])))
                {
                    if (out_queue_size >= HARD_MAX_LENGTH)
                    {
                        errors[b] = 1; // Postfix expression too long
                        break;
                    }
                    out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
                }
                if (op_stack_ptr >= HARD_MAX_LENGTH)
                {
                    errors[b] = 2; // Operator stack overflow
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
                errors[b] = 3; // Mismatched parenthesis
                break;
            }
            if (out_queue_size >= HARD_MAX_LENGTH)
            {
                errors[b] = 1; // Postfix expression too long
                break;
            }
            out_queue[out_queue_size++] = op_stack[--op_stack_ptr];
        }

        if (errors[b] != 0)
            continue;

        for (size_t i = 0; i < out_queue_size; ++i)
        {
            ops_acc[b][i] = out_queue[i];
        }
        for (size_t i = out_queue_size; i < M; ++i)
        {
            ops_acc[b][i] = NO_OP; // padding
        }

        int64_t child_stack[HARD_MAX_LENGTH];
        int64_t child_stack_ptr = 0;
        for (size_t i = 0; i < out_queue_size; i++)
        {
            int64_t token = out_queue[i];
            if (is_unary(token))
            {
                if (child_stack_ptr < 1)
                {
                    errors[b] = 5; // Unary operator without operand
                    break;
                }
                int64_t child_index = child_stack[--child_stack_ptr];
                children_acc[b][i][0] = child_index;
                children_acc[b][i][1] = NULL_CHILD;
            }
            else if (is_binary(token))
            {
                if (child_stack_ptr < 2)
                {
                    errors[b] = 6; // Binary operator without enough operands
                    break;
                }
                int64_t right_child_index = child_stack[--child_stack_ptr];
                int64_t left_child_index = child_stack[--child_stack_ptr];
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
                errors[b] = 7; // Child stack overflow
                break;
            }
            child_stack[child_stack_ptr++] = i;
        }
        if (out_queue_size > 0 && child_stack_ptr != 1)
            errors[b] = 8; // Malformed expression (e.g., too many operands)
    }
}