
#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include <vector>

// Part 4: Operation Numbering and Extensibility
// This enum defines the operator codes used in the `Ops` tensor.
enum Operator : int
{
    // Meta
    NO_OP = 0,
    LEARNABLE_CONSTANT = 1,

    // Hard Constants
    CONST_1 = 2,
    // Add more if needed, e.g., PI, E

    // Unary Functions
    SIN = 6,
    COS = 7,
    EXP = 8,
    LOG = 9,
    SQUARE = 10,
    SQRT = 11,

    // Binary Functions
    ADD = 12,
    SUB = 13,
    MUL = 14,
    DIV = 15,

    // Variable Start ID
    // An op_code >= VAR_START_ID represents variable X_{op_code - VAR_START_ID}
    VAR_START_ID = 16
};

// Compile-time constants
constexpr size_t MAX_ARITY = 2;
constexpr size_t MAX_VARS = 30; // Max number of X variables

C10_HOST_DEVICE inline int get_arity(int op)
{
    if (op >= SIN && op <= SQRT)
        return 1;
    if (op >= ADD && op <= DIV)
        return 2;
    return 0;
}

// --- Wrappers with Strict Domain Checking ---

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t sin_wrapper(scalar_t val)
{
    return sin(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t cos_wrapper(scalar_t val)
{
    return cos(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t exp_wrapper(scalar_t val)
{
    return exp(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t log_wrapper(scalar_t val)
{
    // POLICY: Return NaN for log of non-positive numbers.
    if (val <= static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return log(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t square_wrapper(scalar_t val)
{
    // This wrapper is fine as is.
    return val * val;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t sqrt_wrapper(scalar_t val)
{
    // POLICY: Return NaN for sqrt of negative numbers.
    if (val < static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return sqrt(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t add_wrapper(scalar_t a, scalar_t b)
{
    return a + b;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t sub_wrapper(scalar_t a, scalar_t b)
{
    return a - b;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t mul_wrapper(scalar_t a, scalar_t b)
{
    return a * b;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t div_wrapper(scalar_t a, scalar_t b)
{
    // POLICY: Return NaN on division by zero.
    if (b == static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return a / b;
}