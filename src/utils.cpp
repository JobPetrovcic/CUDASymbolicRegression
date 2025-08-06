#include "operators.h"
#include <pybind11/pybind11.h>

void init_utils(pybind11::module &m)
{
    m.def("get_arity", get_arity, "Get the arity of an operator",
          pybind11::arg("op"))
        .def("is_valid_op", is_valid_op, "Check if the operator is valid based on the number of variables",
             pybind11::arg("op"), pybind11::arg("n_variables"));
}