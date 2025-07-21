#include "operators.h"
#include <pybind11/pybind11.h>

void init_utils(pybind11::module &m)
{
    m.def("get_arity", get_arity, "Get the arity of an operator",
          pybind11::arg("op"));
}