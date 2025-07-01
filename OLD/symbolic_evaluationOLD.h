#pragma once

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
