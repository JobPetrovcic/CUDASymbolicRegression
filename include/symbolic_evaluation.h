#pragma once

#include <torch/extension.h>
#include <vector>
#include "operators.h"
#include "evaluation_kernels.h"

// --- Helper Functions ---
void validate_inputs(const torch::Tensor &Ops, const torch::Tensor &Ch, int n_x);

class SymbolicEvaluation : public torch::autograd::Function<SymbolicEvaluation>
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Ops,
        torch::Tensor Ch,
        torch::Tensor C);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);
};

// Python-facing wrapper function
torch::Tensor evaluate_backend(
    torch::Tensor X,
    torch::Tensor Ops,
    torch::Tensor Ch,
    torch::Tensor C);