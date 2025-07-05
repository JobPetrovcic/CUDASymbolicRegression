#pragma once
#include <torch/extension.h>

// TODO: torch::RestrictPtrTraits

template <typename scalar_t>
void forward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<scalar_t, 2> x_acc,
    torch::TensorAccessor<scalar_t, 1> c_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template <typename scalar_t>
void backward_step_k_cpu_impl(
    torch::TensorAccessor<scalar_t, 3> grad_cache_acc,
    torch::TensorAccessor<scalar_t, 1> grad_c_acc,
    torch::TensorAccessor<scalar_t, 2> grad_x_acc,
    torch::TensorAccessor<scalar_t, 3> cache_acc,
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    torch::TensorAccessor<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

template <typename scalar_t>
void forward_step_k_cuda_impl(
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> x_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> c_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k);

template <typename scalar_t>
void backward_step_k_cuda_impl(
    torch::PackedTensorAccessor64<scalar_t, 3> grad_cache_acc,
    torch::PackedTensorAccessor64<scalar_t, 1> grad_c_acc,
    torch::PackedTensorAccessor64<scalar_t, 2> grad_x_acc,
    torch::PackedTensorAccessor64<scalar_t, 3> cache_acc,
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    torch::PackedTensorAccessor64<int64_t, 2> posC_acc,
    int64_t n_x, int64_t k, int32_t *error_flag_ptr);

void validate_inputs_cuda_impl(
    torch::PackedTensorAccessor64<int64_t, 2> ops_acc,
    torch::PackedTensorAccessor64<int64_t, 3> ch_acc,
    int32_t *error_flag_ptr);

void validate_inputs_cpu_impl(
    torch::TensorAccessor<int64_t, 2> ops_acc,
    torch::TensorAccessor<int64_t, 3> ch_acc,
    int32_t *error_flag_ptr);