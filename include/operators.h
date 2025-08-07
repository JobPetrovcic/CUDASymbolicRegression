#pragma once

#include <cmath>
#include <limits>
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
    TAN = 16,
    ARCSIN = 17,
    ARCCOS = 18,
    ARCTAN = 19,
    SINH = 20,
    COSH = 21,
    TANH = 22,
    FLOOR = 23,
    CEIL = 24,
    LN = 25,     // Natural logarithm (same as LOG but explicit)
    LOG10 = 26,  // Base-10 logarithm
    NEG = 27,    // Unary minus (u-)
    INV = 28,    // Inverse (^-1)
    CUBE = 29,   // ^3
    FOURTH = 30, // ^4
    FIFTH = 31,  // ^5

    // Binary Functions
    ADD = 40,
    SUB = 41,
    MUL = 42,
    DIV = 43,
    POW = 44,

    // Variable Start ID
    // An op_code >= VAR_START_ID represents variable X_{op_code - VAR_START_ID}
    VAR_START_ID = 50
};

// Compile-time constants
constexpr size_t MAX_ARITY = 2;
constexpr size_t MAX_VARS = 30; // Max number of X variables

constexpr int64_t NULL_CHILD = -1;

C10_HOST_DEVICE inline bool is_valid_op(int op, int64_t n_variables)
{
    switch (op)
    {
    case NO_OP:
        return false;
    case LEARNABLE_CONSTANT:
    case CONST_1:
    case CONST_2:
    case CONST_3:
    case CONST_4:
    case CONST_5:
    case PI:
    case E:
    case SIN:
    case COS:
    case EXP:
    case LOG:
    case SQUARE:
    case SQRT:
    case TAN:
    case ARCSIN:
    case ARCCOS:
    case ARCTAN:
    case SINH:
    case COSH:
    case TANH:
    case FLOOR:
    case CEIL:
    case LN:
    case LOG10:
    case NEG:
    case INV:
    case CUBE:
    case FOURTH:
    case FIFTH:
    case ADD:
    case SUB:
    case MUL:
    case DIV:
    case POW:
        return true;
    default:
        // Check if it's a variable
        if (op >= VAR_START_ID && op < VAR_START_ID + n_variables)
        {
            return true;
        }
        return false;
    }
}

C10_HOST_DEVICE inline int get_arity(int op)
{
    if (op >= SIN && op <= FIFTH)
        return 1;
    if (op >= ADD && op <= POW)
        return 2;
    return 0;
}

C10_HOST_DEVICE inline bool is_constant(int op)
{
    return (op >= CONST_1 && op <= E);
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
    case TAN:
    case ARCSIN:
    case ARCCOS:
    case ARCTAN:
    case SINH:
    case COSH:
    case TANH:
    case FLOOR:
    case CEIL:
    case LN:
    case LOG10:
    case NEG:
        return true;
    case INV:
    case SQUARE:
    case CUBE:
    case FOURTH:
    case FIFTH:
        return false;
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
    case TAN:
    case ARCSIN:
    case ARCCOS:
    case ARCTAN:
    case SINH:
    case COSH:
    case TANH:
    case FLOOR:
    case CEIL:
    case LN:
    case LOG10:
    case NEG:
    case INV:
    case CUBE:
    case FOURTH:
    case FIFTH:
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

// --- Additional Unary Function Wrappers ---

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t tan_wrapper(scalar_t val)
{
    return tan(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t arcsin_wrapper(scalar_t val)
{
    // POLICY: Return NaN for values outside [-1, 1]
    if (val < static_cast<scalar_t>(-1.0) || val > static_cast<scalar_t>(1.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return asin(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t arccos_wrapper(scalar_t val)
{
    // POLICY: Return NaN for values outside [-1, 1]
    if (val < static_cast<scalar_t>(-1.0) || val > static_cast<scalar_t>(1.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return acos(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t arctan_wrapper(scalar_t val)
{
    return atan(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t sinh_wrapper(scalar_t val)
{
    return sinh(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t cosh_wrapper(scalar_t val)
{
    return cosh(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t tanh_wrapper(scalar_t val)
{
    return tanh(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t floor_wrapper(scalar_t val)
{
    return floor(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t ceil_wrapper(scalar_t val)
{
    return ceil(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t ln_wrapper(scalar_t val)
{
    // Natural logarithm - same as log_wrapper
    if (val <= static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return log(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t log10_wrapper(scalar_t val)
{
    // Base-10 logarithm
    if (val <= static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return log10(val);
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t neg_wrapper(scalar_t val)
{
    // Unary minus
    return -val;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t inv_wrapper(scalar_t val)
{
    // Inverse (^-1)
    if (val == static_cast<scalar_t>(0.0))
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return static_cast<scalar_t>(1.0) / val;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t cube_wrapper(scalar_t val)
{
    // Cube (^3)
    return val * val * val;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t fourth_wrapper(scalar_t val)
{
    // Fourth power (^4)
    scalar_t sq = val * val;
    return sq * sq;
}

template <typename scalar_t>
C10_HOST_DEVICE inline scalar_t fifth_wrapper(scalar_t val)
{
    // Fifth power (^5)
    scalar_t sq = val * val;
    return sq * sq * val;
}