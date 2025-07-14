#pragma once
#include <torch/torch.h>

constexpr int64_t HARD_MAX_LENGTH = 128;

constexpr int64_t INVALID_RULE = -2;
constexpr int64_t NULL_PARENT = -1;

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
    int64_t B);

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
    const torch::PackedTensorAccessor32<int64_t, 1> seeds,
    int64_t B);

void parse_to_prefix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence,
    const torch::TensorAccessor<int64_t, 2> expressions,
    torch::TensorAccessor<int64_t, 2> ops,
    torch::TensorAccessor<int64_t, 3> children,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void parse_to_prefix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 3> children,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void parse_to_prefix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 2> parents_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M);

void parse_to_prefix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void parse_to_postfix_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence,
    const torch::TensorAccessor<int64_t, 2> expressions,
    torch::TensorAccessor<int64_t, 2> ops,
    torch::TensorAccessor<int64_t, 3> children,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void parse_to_postfix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 3> children,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void parse_to_postfix_parent_cpu_impl(
    const torch::TensorAccessor<int64_t, 1> precedence_acc,
    const torch::TensorAccessor<int64_t, 2> expressions_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 2> parents_acc,
    torch::TensorAccessor<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B, int64_t M);

void parse_to_postfix_parent_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 1> precedence,
    const torch::PackedTensorAccessor32<int64_t, 2> expressions,
    torch::PackedTensorAccessor32<int64_t, 2> ops,
    torch::PackedTensorAccessor32<int64_t, 2> parents,
    torch::PackedTensorAccessor32<int64_t, 1> errors,
    int64_t lparenthesis_id,
    int64_t rparenthesis_id,
    int64_t B,
    int64_t M);

void postfix_to_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> postfix_acc,
    torch::TensorAccessor<int64_t, 2> infix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    const std::unordered_map<int64_t, std::string> &id_to_symbol,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix);

void postfix_to_infix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> postfix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_postfix, int64_t M_infix);

void prefix_to_infix_cpu_impl(
    const torch::TensorAccessor<int64_t, 2> prefix_acc,
    torch::TensorAccessor<int64_t, 2> infix_acc,
    torch::TensorAccessor<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_prefix, int64_t M_infix);

void prefix_to_infix_cuda_impl(
    const torch::PackedTensorAccessor32<int64_t, 2> prefix_acc,
    torch::PackedTensorAccessor32<int64_t, 2> infix_acc,
    torch::PackedTensorAccessor32<int64_t, 1> errors_acc,
    int64_t lparen_id, int64_t rparen_id,
    int64_t B, int64_t M_prefix, int64_t M_infix);