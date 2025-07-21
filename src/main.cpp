#include <torch/extension.h>
#include <pybind11/pybind11.h>

void init_pcfg(pybind11::module &m);
void init_symbolic_evaluation(pybind11::module &m);
void init_utils(pybind11::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    init_pcfg(m);
    init_symbolic_evaluation(m);
    init_utils(m);
}
