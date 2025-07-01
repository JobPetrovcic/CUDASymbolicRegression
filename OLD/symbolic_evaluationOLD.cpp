#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "symbolic_evaluation.h"
#include "kernels.h"

// --- Helper Functions ---
void validate_inputs(const torch::Tensor &Ops, const torch::Tensor &Ch, int n_x)
{
    if (Ops.device() != torch::kCUDA || Ch.device() != torch::kCUDA)
    {
        throw std::runtime_error("Ops and Ch tensors must be on CUDA device.");
    }

    size_t M = Ops.size(0);
    size_t B = Ops.size(1);

    // --- Allocate error flag on GPU ---
    auto options = torch::TensorOptions().dtype(torch::kInt).device(device);
    auto error_flag_tensor = torch::zeros({1}, options);
    int *error_flag_ptr = error_flag_tensor.data_ptr<int>();

    // --- Launch validation kernel ---
    launch_validate_inputs(Ops.data_ptr<int>(), Ch.data_ptr<int>(), M, B, error_flag_ptr);

    // --- Copy error flag to CPU and check ---
    int error_flag_cpu = error_flag_tensor.item<int>();
    AT_ASSERTM(error_flag_cpu == 0, "Input validation failed on CUDA device.");
}

class SymbolicEvaluation : public torch::autograd::Function<SymbolicEvaluation>
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Ops,
        torch::Tensor Ch,
        torch::Tensor C,
        torch::Tensor posC)
    {
        // --- Move inputs to CUDA device ---
        auto device = torch::kCUDA;
        // Ensure tensors are on the correct device
        if (Ops.device() != device || Ch.device() != device || C.device() != device || posC.device() != device)
        {
            throw std::runtime_error("Ops, Ch, C, and posC tensors must be on CUDA device.");
        }

        // --- Get dimensions ---
        const size_t M = Ops.size(0);
        const size_t B = Ops.size(1);
        const int N = X.size(0);
        const int n_x = X.size(1);

        // Part 2.1: Allocate cache
        auto cache = torch::zeros({M, B, N}, X.options());

        // --- Get data pointers ---
        float *cache_ptr = cache.data_ptr<float>();
        const int *ops_ptr = Ops.data_ptr<int>();
        const int *ch_ptr = Ch.data_ptr<int>();
        const float *x_ptr = X.data_ptr<float>();
        const float *Constants_ptr = C.data_ptr<float>();
        const int *ConstantPosition_ptr = posC.data_ptr<int>();

        // Part 2.2: Host-side loop
        for (int k = 0; k < M; ++k)
        {
            launch_forward_step_k(cache_ptr, ops_ptr, ch_ptr, x_ptr, Constants_ptr, ConstantPosition_ptr, M, B, N, n_x, k);
        }

        // --- Save tensors for backward pass ---
        ctx->save_for_backward({X, Ops, Ch, C, posC, cache});

        // Part 2.4: Return final result
        return cache;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto X = saved[0];
        auto Ops = saved[1];
        auto Ch = saved[2];
        auto C = saved[3];
        auto posC = saved[4];
        auto cache = saved[5];

        // assert device is CUDA
        if (X.device() != torch::kCUDA || Ops.device() != torch::kCUDA || Ch.device() != torch::kCUDA || C.device() != torch::kCUDA || posC.device() != torch::kCUDA || cache.device() != torch::kCUDA)
        {
            throw std::runtime_error("X, Ops, Ch, C, posC, and cache tensors must be on CUDA device, but got " +
                                     X.device().str() + ", " + Ops.device().str() + ", " + Ch.device().str() +
                                     ", " + C.device().str() + ", " + posC.device().str() + ", " + cache.device().str());
        }

        auto grad_output = grad_outputs[0].contiguous();

        // --- Get dimensions ---
        const size_t M = Ops.size(0);
        const size_t B = Ops.size(1);
        const int N = X.size(0);
        const int n_x = X.size(1);
        // const int SC = C.size(0);

        // Part 3.1: Allocate gradient tensors
        auto grad_cache = torch::zeros_like(cache);
        auto grad_C = torch::zeros_like(C);
        auto grad_X = torch::zeros_like(X);

        // Part 3.2: Initialize grad_cache
        grad_cache.slice(0, M - 1, M).copy_(grad_output.unsqueeze(0));

        // --- Get data pointers ---
        float *grad_cache_ptr = grad_cache.data_ptr<float>();
        float *grad_Constants_ptr = grad_C.data_ptr<float>();
        float *grad_x_ptr = grad_X.data_ptr<float>();
        const float *cache_ptr = cache.data_ptr<float>();
        const int *ops_ptr = Ops.data_ptr<int>();
        const int *ch_ptr = Ch.data_ptr<int>();
        const int *ConstantPosition_ptr = posC.data_ptr<int>();

        // Part 3.3: Host-side loop
        for (int k = M - 1; k >= 0; --k)
        {
            launch_backward_step_k(
                grad_cache_ptr, grad_Constants_ptr, grad_x_ptr,
                cache_ptr, ops_ptr, ch_ptr, ConstantPosition_ptr,
                M, B, N, n_x, k);
        }

        // Return gradients for X and C. Grads for Ops, Ch, posC are None.
        return {grad_X, torch::Tensor(), torch::Tensor(), grad_C, torch::Tensor()};
    }
};

// Python-facing wrapper function
std::tuple<torch::Tensor, torch::Tensor> evaluate(
    torch::Tensor X,
    torch::Tensor Ops,
    torch::Tensor Ch,
    c10::optional<torch::Tensor> C_opt)
{
    // --- Part 1: Input Validation ---
    int n_x = X.size(1);
    AT_ASSERTM(Ops.size(1) == Ch.size(1), "Ops and Ch must have the same batch size (dim 1)");
    AT_ASSERTM(Ops.size(0) == Ch.size(0), "Ops and Ch must have the same max length (dim 0)");
    AT_ASSERTM(Ch.size(2) == MAX_ARITY, "Ch must have MaxArity as its last dimension");

    // assert all tensors are on the CUDA device
    if (X.device() != torch::kCUDA || Ops.device() != torch::kCUDA || Ch.device() != torch::kCUDA ||
        (C_opt.has_value() && C_opt->device() != torch::kCUDA))
    {
        throw std::runtime_error("X, Ops, and Ch tensors must be on CUDA device, but got " + X.device().str() + ", " + Ops.device().str() + ", " + Ch.device().str());
    }

    validate_inputs(Ops, Ch, n_x);

    torch::Tensor C;

    // --- Part 1: Constant Handling and posC Tensor Creation ---
    // Scan Ops tensor column-by-column to find learnable constants
    auto ops_transposed = Ops.transpose(0, 1).contiguous();
    auto const_indices = (ops_transposed == LEARNABLE_CONSTANT).nonzero();
    int SC = const_indices.size(0);

    if (C_opt)
    {
        C = *C_opt;
        AT_ASSERTM(C.dim() == 1 && C.size(0) == SC, "Provided C has incorrect size.");
    }
    else
    {
        // Create a new tensor on CUDA device
        C = torch::randn({SC}, Ops.options().dtype(torch::kFloat).device(torch::kCUDA));
    }

    // assert that C is on the CUDA device
    if (C.device() != torch::kCUDA)
    {
        throw std::runtime_error("C tensor must be on CUDA device, but got " +
                                 C.device().str());
    }

    auto posC = torch::full_like(Ops, -1, Ops.options().dtype(torch::kInt));
    if (SC > 0)
    {
        // Need to map flat indices back to 2D
        auto const_indices_2d = torch::stack({const_indices.select(1, 1), const_indices.select(1, 0)}, 1);
        posC.index_put_({const_indices_2d.select(1, 0), const_indices_2d.select(1, 1)}, torch::arange(SC, Ops.options().dtype(torch::kInt)));
    }

    auto result = SymbolicEvaluation::apply(X, Ops, Ch, C, posC);

    return std::make_tuple(result, C);
}

// --- Pybind11 Bindings ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<Operator>(m, "Operator")
        .value("NO_OP", NO_OP)
        .value("LEARNABLE_CONSTANT", LEARNABLE_CONSTANT)
        .value("CONST_1", CONST_1)
        .value("SIN", SIN)
        .value("COS", COS)
        .value("EXP", EXP)
        .value("LOG", LOG)
        .value("SQUARE", SQUARE)
        .value("SQRT", SQRT)
        .value("ADD", ADD)
        .value("SUB", SUB)
        .value("MUL", MUL)
        .value("DIV", DIV)
        .value("VAR_START_ID", VAR_START_ID)
        .export_values();

    m.def("evaluate", &evaluate, "Evaluate symbolic expressions with CUDA backend",
          py::arg("X"), py::arg("Ops"), py::arg("Ch"), py::arg("C") = py::none());
}