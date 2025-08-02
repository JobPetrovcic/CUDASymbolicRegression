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
    torch::Tensor X,   // (N, n_x)
    torch::Tensor Ops, // (M, B)
    torch::Tensor Ch,  // (M, B, MAX_ARITY)
    torch::Tensor C);  // (M, B)

class SymbolicEvaluationMultiple : public torch::autograd::Function<SymbolicEvaluationMultiple>
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

torch::Tensor evaluate_multiple_constant_backend(
    torch::Tensor X,   // (N, n_x)
    torch::Tensor Ops, // (M, B)
    torch::Tensor Ch,  // (M, B, MAX_ARITY)
    torch::Tensor C);  // (M, B, K)
