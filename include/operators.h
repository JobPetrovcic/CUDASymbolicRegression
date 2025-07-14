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
    CONST_2 = 3,
    CONST_3 = 4,
    CONST_4 = 5,
    CONST_5 = 6,
    PI = 7,
    E = 8,

    // Unary Functions
    SIN = 10,
    COS = 11,
    EXP = 12,
    LOG = 13,
    SQUARE = 14,
    SQRT = 15,

    // Binary Functions
    ADD = 20,
    SUB = 21,
    MUL = 22,
    DIV = 23,
    POW = 24,

    // Variable Start ID
    // An op_code >= VAR_START_ID represents variable X_{op_code - VAR_START_ID}
    VAR_START_ID = 30
};

// Compile-time constants
constexpr size_t MAX_ARITY = 2;
constexpr size_t MAX_VARS = 30; // Max number of X variables

constexpr int64_t NULL_CHILD = -1;

C10_HOST_DEVICE inline int get_arity(int op)
{
    if (op >= SIN && op <= SQRT)
        return 1;
    if (op >= ADD && op <= POW)
        return 2;
    return 0;
}

C10_HOST_DEVICE inline bool is_constant(int op)
{
    return (op >= CONST_1 && op < SIN);
}

C10_HOST_DEVICE inline bool is_unary(int op)
{
    return get_arity(op) == 1;
}

// Helper function to determine if an operator is functional style (e.g., sin(x))
C10_HOST_DEVICE inline bool is_functional_style(int64_t op_id)
{
    switch (op_id)
    {
    case SIN:
    case COS:
    case EXP:
    case LOG:
    case SQRT:
        return true;
    default:
        return false;
    }
}

C10_HOST_DEVICE inline bool is_binary(int op)
{
    return get_arity(op) == 2;
}

C10_HOST_DEVICE inline bool is_right_associative(int op)
{
    // Returns true if the operator is right-associative, false if left-associative
    switch (op)
    {
    case ADD:
    case SUB:
    case MUL:
    case DIV:
        return false; // Left associative
    case POW:
    case SIN:
    case COS:
    case EXP:
    case LOG:
    case SQUARE:
    case SQRT:
        return true; // Right associative
    default:
        return false; // Default to left associative for unknown ops
    }
}

C10_HOST_DEVICE inline bool is_left_associative(int op)
{
    return !is_right_associative(op);
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

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t pow_wrapper(scalar_t a, scalar_t b)
{
    if (a < static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN(); // POLICY: Return NaN for negative base
    }
    return pow(a, b);
}