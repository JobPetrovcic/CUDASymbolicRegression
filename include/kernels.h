#pragma once
#include <torch/extension.h>

template <typename scalar_t>
void forward_step_k_cpu_impl(
    scalar_t *cache_ptr,
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    const scalar_t *x_ptr,
    const scalar_t *c_ptr,
    const int64_t *ConstantPosition_ptr,
    int64_t M, int64_t B, int64_t N, int64_t n_x,
    int64_t k);

template <typename scalar_t>
void backward_step_k_cpu_impl(
    scalar_t *grad_cache_ptr,
    scalar_t *grad_c_ptr,
    scalar_t *grad_x_ptr,
    const scalar_t *cache_ptr,
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    const int64_t *ConstantPosition_ptr,
    int64_t M, int64_t B, int64_t N, int64_t n_x,
    int64_t k);

template <typename scalar_t>
void forward_step_k_cuda_impl(
    scalar_t *cache_ptr,
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    const scalar_t *x_ptr,
    const scalar_t *c_ptr,
    const int64_t *ConstantPosition_ptr,
    int64_t M, int64_t B, int64_t N, int64_t n_x,
    int64_t k);

template <typename scalar_t>
void backward_step_k_cuda_impl(
    scalar_t *grad_cache_ptr,
    scalar_t *grad_c_ptr,
    scalar_t *grad_x_ptr,
    const scalar_t *cache_ptr,
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    const int64_t *ConstantPosition_ptr,
    int64_t M, int64_t B, int64_t N, int64_t n_x,
    int64_t k);

void validate_inputs_cuda_impl(
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    int32_t *error_flag_ptr,
    int64_t M, int64_t B);

void validate_inputs_cpu_impl(
    const int64_t *ops_ptr,
    const int64_t *ch_ptr,
    int32_t *error_flag_ptr,
    int64_t M, int64_t B);