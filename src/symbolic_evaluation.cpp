#include "symbolic_evaluation.h"
#include <iostream>
#include <c10/cuda/CUDAStream.h>

// --- Helper Functions ---
void validate_inputs(const torch::Tensor &Ops, const torch::Tensor &Ch, int n_x)
{
    const auto device = Ops.device();
    TORCH_CHECK(device == Ch.device(), "Ops and Ch must be on the same device");
    // Check contiguity
    TORCH_CHECK(Ops.is_contiguous(), "Ops tensor must be contiguous");
    TORCH_CHECK(Ch.is_contiguous(), "Ch tensor must be contiguous");

    // --- Allocate error flag on the same device ---
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto error_flag_tensor = torch::zeros({1}, options);
    int32_t *error_flag_ptr = error_flag_tensor.data_ptr<int32_t>();

    // --- Launch validation kernel ---
    if (device.is_cuda())
    {
        validate_inputs_cuda_impl(Ops.packed_accessor64<int64_t, 2>(), Ch.packed_accessor64<int64_t, 3>(), error_flag_ptr);
    }
    else
    {
        validate_inputs_cpu_impl(Ops.accessor<int64_t, 2>(), Ch.accessor<int64_t, 3>(), error_flag_ptr);
    }

    int32_t error_flag_cpu = error_flag_tensor.item<int32_t>();
    TORCH_CHECK(error_flag_cpu == 0, "Input validation failed. This may be due to an invalid operator or child index.");
}

torch::Tensor SymbolicEvaluation::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor Ops,
    torch::Tensor Ch,
    torch::Tensor C,
    torch::Tensor posC)
{
    const auto device = X.device();
    // Ensure tensors are on the correct device
    TORCH_INTERNAL_ASSERT(Ops.device() == device && Ch.device() == device && C.device() == device && posC.device() == device,
                          "All input tensors must be on the same device.");
    // Ensure tensors are contiguous
    TORCH_INTERNAL_ASSERT(Ops.is_contiguous() && Ch.is_contiguous() && C.is_contiguous() && posC.is_contiguous(),
                          "All input tensors must be contiguous.");

    // --- Get dimensions ---
    const int64_t M = Ops.size(0);
    const int64_t B = Ops.size(1);
    const int64_t N = X.size(0);
    const int64_t n_x = X.size(1);

    auto cache = torch::zeros({M, N, B}, X.options());

    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "symbolic_forward", ([&]
                                                                     {
        if (device.is_cuda())
        {
            auto cache_acc = cache.packed_accessor64<scalar_t, 3>();
            auto ops_acc = Ops.packed_accessor64<int64_t, 2>();
            auto ch_acc = Ch.packed_accessor64<int64_t, 3>();
            auto x_acc = X.packed_accessor64<scalar_t, 2>();
            auto c_acc = C.packed_accessor64<scalar_t, 1>();
            auto posC_acc = posC.packed_accessor64<int64_t, 2>();
            for (int64_t k = 0; k < M; ++k)
            {
                evaluation_forward_step_k_cuda_impl<scalar_t>(cache_acc, ops_acc, ch_acc, x_acc, c_acc, posC_acc, n_x, k);
            }
        }
        else
        {
            auto cache_acc = cache.accessor<scalar_t, 3>();
            auto ops_acc = Ops.accessor<int64_t, 2>();
            auto ch_acc = Ch.accessor<int64_t, 3>();
            auto x_acc = X.accessor<scalar_t, 2>();
            auto c_acc = C.accessor<scalar_t, 1>();
            auto posC_acc = posC.accessor<int64_t, 2>();
            for (int64_t k = 0; k < M; ++k)
            {
                evaluation_forward_step_k_cpu_impl<scalar_t>(cache_acc, ops_acc, ch_acc, x_acc, c_acc, posC_acc, n_x, k);
            }
        } }));

    ctx->save_for_backward({X, Ops, Ch, C, posC, cache});

    return cache;
}

torch::autograd::variable_list SymbolicEvaluation::backward(
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

    const auto device = X.device();
    const int64_t M = Ops.size(0);
    const int64_t n_x = X.size(1);
    const int64_t SC = C.size(0);

    auto grad_output = grad_outputs[0];
    TORCH_INTERNAL_ASSERT(cache.sizes() == grad_output.sizes(),
                          "Cache and grad_output must have the same shape. "
                          "Cache shape: ",
                          cache.sizes(), ", grad_output shape: ", grad_output.sizes());
    auto grad_cache = torch::zeros_like(cache);
    grad_cache.add_(grad_output);

    auto grad_X = torch::zeros_like(X);
    auto grad_C = torch::zeros_like(C);

    // --- Allocate error flag on the same device ---
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto error_flag_tensor = torch::zeros({1}, options);
    int32_t *error_flag_ptr = error_flag_tensor.data_ptr<int32_t>();

    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "symbolic_backward", ([&]
                                                                      {
        if (device.is_cuda())
        {
            auto grad_cache_acc = grad_cache.packed_accessor64<scalar_t, 3>();
            auto grad_C_acc = grad_C.packed_accessor64<scalar_t, 1>();
            auto grad_X_acc = grad_X.packed_accessor64<scalar_t, 2>();
            auto cache_acc = cache.packed_accessor64<scalar_t, 3>();
            auto ops_acc = Ops.packed_accessor64<int64_t, 2>();
            auto ch_acc = Ch.packed_accessor64<int64_t, 3>();
            auto posC_acc = posC.packed_accessor64<int64_t, 2>();
            for (int64_t k = M - 1; k >= 0; --k)
            {
                evaluation_backward_step_k_cuda_impl<scalar_t>(grad_cache_acc, grad_C_acc, grad_X_acc, cache_acc, ops_acc, ch_acc, posC_acc, n_x, k, error_flag_ptr);
            }
        }
        else
        {
            auto grad_cache_acc = grad_cache.accessor<scalar_t, 3>();
            auto grad_C_acc = grad_C.accessor<scalar_t, 1>();
            auto grad_X_acc = grad_X.accessor<scalar_t, 2>();
            auto cache_acc = cache.accessor<scalar_t, 3>();
            auto ops_acc = Ops.accessor<int64_t, 2>();
            auto ch_acc = Ch.accessor<int64_t, 3>();
            auto posC_acc = posC.accessor<int64_t, 2>();
            for (int64_t k = M - 1; k >= 0; --k)
            {
                evaluation_backward_step_k_cpu_impl<scalar_t>(grad_cache_acc, grad_C_acc, grad_X_acc, cache_acc, ops_acc, ch_acc, posC_acc, n_x, k, error_flag_ptr);
            }
        } }));

    int32_t error_flag_cpu = error_flag_tensor.item<int32_t>();

    // Check for errors from the backward pass
    if (error_flag_cpu != 0)
    {
        switch (error_flag_cpu)
        {
        case 1:
            TORCH_CHECK(false, "Backward error: NaN/inf gradient detected. Error code 1.");
            break;
        case 2:
            TORCH_CHECK(false, "Backward error: Gradient propagated to a NO_OP node. Error code 2.");
            break;
        case 3:
            TORCH_CHECK(false, "Backward error: Gradient propagated to a LOG node with a non-positive argument. Error code 3.");
            break;
        case 4:
            TORCH_CHECK(false, "Backward error: Gradient propagated to a SQRT node with a non-positive argument. Error code 4.");
            break;
        case 5:
            TORCH_CHECK(false, "Backward error: Gradient propagated to a DIV node with a zero denominator. Error code 5.");
            break;
        default:
            TORCH_CHECK(false, "Unknown backward error. Error code: ", error_flag_cpu);
            break;
        }
    }

    return {grad_X,
            torch::Tensor(),
            torch::Tensor(),
            grad_C,
            torch::Tensor()};
}

// Python-facing wrapper function
torch::Tensor evaluate_backend(
    torch::Tensor X,   // (N, n_x)
    torch::Tensor Ops, // (M, B)
    torch::Tensor Ch,  // (M, B, MAX_ARITY)
    torch::Tensor C    // (SC)
)
{
    const auto device = X.device();
    TORCH_CHECK(Ops.device() == device && Ch.device() == device && C.device() == device,
                "All input tensors must be on the same device.");

    const int n_x = X.size(1);
    TORCH_CHECK(Ops.dim() == 2, "Ops must be a 2D tensor");
    TORCH_CHECK(Ch.dim() == 3, "Ch must be a 3D tensor");
    TORCH_CHECK(Ops.size(0) == Ch.size(0), "Ops and Ch must have the same max length (dim 0)");
    TORCH_CHECK(Ops.size(1) == Ch.size(1), "Ops and Ch must have the same batch size (dim 1)");
    TORCH_CHECK(Ch.size(2) == MAX_ARITY, "Ch must have MaxArity as its last dimension");

    // Check that X and C are floating and Ch and Ops are Long tensors
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kDouble,
                "X must be a floating-point tensor (float or double)");
    TORCH_CHECK(C.scalar_type() == torch::kFloat || C.scalar_type() == torch::kDouble,
                "C must be a floating-point tensor (float or double)");

    TORCH_CHECK(Ops.scalar_type() == torch::kInt64, "Ops must be a Long tensor");
    TORCH_CHECK(Ch.scalar_type() == torch::kInt64, "Ch must be a Long tensor");

    // If the tensors are not contiguous, we need to make them contiguous
    if (!Ops.is_contiguous())
    {
        Ops = Ops.contiguous();
    }
    if (!Ch.is_contiguous())
    {
        Ch = Ch.contiguous();
    }
    if (!C.is_contiguous())
    {
        C = C.contiguous();
    }

    validate_inputs(Ops, Ch, n_x);

    int64_t SC = 0;

    // --- Part 1: Constant Handling and posC Tensor Creation ---
    // Scan Ops tensor to find learnable constants
    auto const_indices = (Ops == LEARNABLE_CONSTANT).nonzero();
    SC = const_indices.size(0);

    TORCH_CHECK(C.dim() == 1, "C must be a 1D tensor");
    TORCH_CHECK(C.size(0) == SC, "The size of C must match the number of learnable constants in Ops");

    auto posC = torch::zeros_like(Ops, Ops.options().dtype(torch::kInt64));
    if (SC > 0)
    {
        posC.index_put_({const_indices.select(1, 0), const_indices.select(1, 1)},
                        torch::arange(SC, Ops.options().dtype(torch::kInt64)));
    }

    auto result = SymbolicEvaluation::apply(X, Ops, Ch, C, posC);

    return result;
}

void init_symbolic_evaluation(pybind11::module &m)
{
    pybind11::enum_<Operator>(m, "Operator", "Enum for symbolic operations", py::arithmetic())
        .value("NO_OP", Operator::NO_OP)
        .value("LEARNABLE_CONSTANT", Operator::LEARNABLE_CONSTANT)
        .value("CONST_1", Operator::CONST_1)
        .value("CONST_2", Operator::CONST_2)
        .value("CONST_3", Operator::CONST_3)
        .value("CONST_4", Operator::CONST_4)
        .value("CONST_5", Operator::CONST_5)
        .value("PI", Operator::PI)
        .value("E", Operator::E)
        .value("SIN", Operator::SIN)
        .value("COS", Operator::COS)
        .value("EXP", Operator::EXP)
        .value("LOG", Operator::LOG)
        .value("SQUARE", Operator::SQUARE)
        .value("SQRT", Operator::SQRT)
        .value("ADD", Operator::ADD)
        .value("SUB", Operator::SUB)
        .value("MUL", Operator::MUL)
        .value("DIV", Operator::DIV)
        .value("POW", Operator::POW)
        .value("VAR_START_ID", Operator::VAR_START_ID)
        .export_values();

    m.def("evaluate_backend", static_cast<torch::Tensor (*)(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>(&evaluate_backend), "Evaluate a batch of symbolic expressions.");
}