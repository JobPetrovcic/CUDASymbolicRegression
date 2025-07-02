#include "symbolic_evaluation.h"

// --- Helper Functions ---
void validate_inputs(const torch::Tensor &Ops, const torch::Tensor &Ch, int n_x)
{
    const auto device = Ops.device();
    TORCH_CHECK(device == Ch.device(), "Ops and Ch must be on the same device");
    // Check contiguity
    TORCH_CHECK(Ops.is_contiguous(), "Ops tensor must be contiguous");
    TORCH_CHECK(Ch.is_contiguous(), "Ch tensor must be contiguous");

    size_t M = Ops.size(0);
    size_t B = Ops.size(1);

    // --- Allocate error flag on the same device ---
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto error_flag_tensor = torch::zeros({1}, options);
    int32_t *error_flag_ptr = error_flag_tensor.data_ptr<int32_t>();

    // --- Launch validation kernel ---
    if (device.is_cuda())
    {
        validate_inputs_cuda_impl(Ops.data_ptr<int64_t>(), Ch.data_ptr<int64_t>(), error_flag_ptr, M, B);
    }
    else
    {
        validate_inputs_cpu_impl(Ops.data_ptr<int64_t>(), Ch.data_ptr<int64_t>(), error_flag_ptr, M, B);
    }

    // --- Copy error flag to CPU and check ---
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

    auto cache = torch::zeros({M, B, N}, X.options());

    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "symbolic_forward", ([&]
                                                                     {
        scalar_t *cache_ptr = cache.data_ptr<scalar_t>();
        const int64_t *ops_ptr = Ops.data_ptr<int64_t>();
        const int64_t *ch_ptr = Ch.data_ptr<int64_t>();
        const scalar_t *x_ptr = X.data_ptr<scalar_t>();
        const scalar_t *Constants_ptr = C.data_ptr<scalar_t>();
        const int64_t *ConstantPosition_ptr = posC.data_ptr<int64_t>();

        for (int64_t k = 0; k < M; ++k)
        {
            if (device.is_cuda())
            {
                forward_step_k_cuda_impl<scalar_t>(cache_ptr, ops_ptr, ch_ptr, x_ptr, Constants_ptr, ConstantPosition_ptr, M, B, N, n_x, k);
            }
            else
            {
                forward_step_k_cpu_impl<scalar_t>(cache_ptr, ops_ptr, ch_ptr, x_ptr, Constants_ptr, ConstantPosition_ptr, M, B, N, n_x, k);
            }
        } }));

    ctx->save_for_backward({X, Ops, Ch, C, posC, cache});

    return cache.slice(0, M - 1, M).squeeze(0);
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
    const int64_t B = Ops.size(1);
    const int64_t N = X.size(0);
    const int64_t n_x = X.size(1);
    const int64_t SC = C.size(0);

    auto grad_output = grad_outputs[0];
    auto grad_cache = torch::zeros_like(cache);
    grad_cache.slice(0, M - 1, M) = grad_output.unsqueeze(0);

    auto grad_X = torch::zeros_like(X);
    auto grad_C = torch::zeros_like(C);

    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "symbolic_backward", ([&]
                                                                      {
        scalar_t *grad_cache_ptr = grad_cache.data_ptr<scalar_t>();
        scalar_t *grad_c_ptr = grad_C.data_ptr<scalar_t>();
        scalar_t *grad_x_ptr = grad_X.data_ptr<scalar_t>();
        const scalar_t *cache_ptr = cache.data_ptr<scalar_t>();
        const int64_t *ops_ptr = Ops.data_ptr<int64_t>();
        const int64_t *ch_ptr = Ch.data_ptr<int64_t>();
        const int64_t *ConstantPosition_ptr = posC.data_ptr<int64_t>();

        for (int64_t k = M - 1; k >= 0; --k)
        {
            if (device.is_cuda())
            {
                backward_step_k_cuda_impl<scalar_t>(grad_cache_ptr, grad_c_ptr, grad_x_ptr, cache_ptr, ops_ptr, ch_ptr, ConstantPosition_ptr, M, B, N, n_x, k);
            }
            else
            {
                backward_step_k_cpu_impl<scalar_t>(grad_cache_ptr, grad_c_ptr, grad_x_ptr, cache_ptr, ops_ptr, ch_ptr, ConstantPosition_ptr, M, B, N, n_x, k);
            }
        } }));

    return {grad_X,
            torch::Tensor(),
            torch::Tensor(),
            grad_C,
            torch::Tensor()};
}

// Python-facing wrapper function
std::tuple<torch::Tensor, torch::Tensor> evaluate(
    torch::Tensor X,
    torch::Tensor Ops,
    torch::Tensor Ch,
    c10::optional<torch::Tensor> C_opt)
{
    const auto device = X.device();
    TORCH_CHECK(Ops.device() == device && Ch.device() == device &&
                    (!C_opt.has_value() || C_opt->device() == device),
                "All input tensors must be on the same device.");

    const int n_x = X.size(1);
    TORCH_CHECK(Ops.dim() == 2, "Ops must be a 2D tensor");
    TORCH_CHECK(Ch.dim() == 3, "Ch must be a 3D tensor");
    TORCH_CHECK(Ops.size(0) == Ch.size(0), "Ops and Ch must have the same max length (dim 0)");
    TORCH_CHECK(Ops.size(1) == Ch.size(1), "Ops and Ch must have the same batch size (dim 1)");
    TORCH_CHECK(Ch.size(2) == MAX_ARITY, "Ch must have MaxArity as its last dimension");
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

    validate_inputs(Ops, Ch, n_x);

    torch::Tensor C;
    int64_t SC = 0;

    // --- Part 1: Constant Handling and posC Tensor Creation ---
    // Scan Ops tensor to find learnable constants
    auto const_indices = (Ops == LEARNABLE_CONSTANT).nonzero();
    SC = const_indices.size(0);

    if (C_opt)
    {
        C = *C_opt;
        TORCH_CHECK(C.dim() == 1, "C must be a 1D tensor");
        TORCH_CHECK(C.size(0) == SC, "The size of C must match the number of learnable constants in Ops");
    }
    else
    {
        C = torch::randn({SC}, X.options());
    }

    auto posC = torch::full_like(Ops, -1, Ops.options().dtype(torch::kInt64));
    if (SC > 0)
    {
        posC.index_put_({const_indices.select(1, 0), const_indices.select(1, 1)},
                        torch::arange(SC, Ops.options().dtype(torch::kInt64)));
    }

    auto result = SymbolicEvaluation::apply(X, Ops, Ch, C, posC);

    return std::make_tuple(result, C);
}

// --- Pybind11 Bindings ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("evaluate", &evaluate, "Evaluate a batch of symbolic expressions.");
}