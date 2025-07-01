#pragma once
#include <torch/extension.h>

// Note: These functions are implemented separately for CPU and CUDA.
// The dispatcher will call the correct one based on the tensor's device.

void forward_step_k_impl(
    torch::Tensor cache,
    const torch::Tensor &ops,
    const torch::Tensor &ch,
    const torch::Tensor &x,
    const torch::Tensor &Constants,
    const torch::Tensor &ConstantPosition,
    int64_t k);

void backward_step_k_impl(
    torch::Tensor grad_cache,
    torch::Tensor grad_Constants,
    torch::Tensor grad_x,
    const torch::Tensor &cache,
    const torch::Tensor &ops,
    const torch::Tensor &ch,
    const torch::Tensor &ConstantPosition,
    int64_t k);

void validate_inputs_impl(
    const torch::Tensor &ops,
    const torch::Tensor &ch,
    torch::Tensor &error_flag);