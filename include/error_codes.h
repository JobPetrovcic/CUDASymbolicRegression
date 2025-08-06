#pragma once

#include <cstdint>
#include <string>
#include <sstream>

// This file defines specific error codes used throughout the symbolic-torch library.
// The integer values are dispersed to allow for future additions within each category
// without requiring renumbering. Each error code should be unique to pinpoint the
// exact location and cause of a failure.
//
// The underlying type is int64_t to match the error tensors used in PCFG operations.

enum class ErrorCode : int64_t
{
    // --- General ---
    NO_ERROR = 0,

    // --- Input Validation Errors (from validate_inputs) ---
    VALIDATION_ERROR_START = 100,
    VALIDATION_CHILD_INDEX_GEQ_PARENT = 101, // In postfix/prefix, a child's index must be less than its parent's.
    VALIDATION_ARITY_MISMATCH = 102,         // The number of non-null children does not match the operator's expected arity.

    // --- Symbolic Evaluation Backward Pass Errors ---
    EVAL_BACKWARD_ERROR_START = 200,
    EVAL_BACKWARD_GRAD_ON_NO_OP = 201, // A non-zero gradient was passed to a NO_OP node.
                                       // EVAL_BACKWARD_LOG_AT_NON_POSITIVE = 202, // Gradient for log(x) where x <= 0. // Not used in Pytorch, so commented out.
                                       // EVAL_BACKWARD_SQRT_AT_NEGATIVE = 203,       // Gradient for sqrt(x) where x < 0. Not used in Pytorch.
    EVAL_BACKWARD_DIV_BY_ZERO = 204,   // Gradient for a/b where b = 0.
                                       // EVAL_BACKWARD_POW_WITH_NEGATIVE_BASE = 205, // Gradient for pow(a, b) where a < 0. Not used in Pytorch.

    // --- PCFG Generation/Sampling Errors (from pcfg_sample_string_expression) ---
    GENERATION_ERROR_START = 300,
    GENERATION_STACK_OVERFLOW = 301,      // The internal generation stack exceeded its capacity.
    GENERATION_EXPRESSION_TOO_LONG = 302, // The generated string of tokens exceeded the maximum padded length.
    GENERATION_MAX_TRIES_EXCEEDED = 303,  // Failed to generate a valid expression within the `max_tries` limit.
    GENERATION_NO_VALID_RULE_FOUND = 304, // During sampling, a random number did not map to any rule (should be impossible if probabilities sum to 1).

    // --- Infix-to-Postfix/Prefix Parsing Errors (from parse_to_postfix/prefix) ---
    PARSING_ERROR_START = 400,
    PARSING_OUTPUT_QUEUE_OVERFLOW = 401,           // The output queue for the shunting-yard algorithm exceeded its capacity.
    PARSING_OPERATOR_STACK_OVERFLOW = 402,         // The operator stack in the shunting-yard algorithm overflowed.
    PARSING_MISMATCHED_PARENTHESIS = 403,          // Unbalanced or mismatched parentheses detected.
    PARSING_TREE_UNARY_OP_MISSING_OPERAND = 404,   // During tree construction from postfix/prefix, a unary operator was found without a valid operand.
    PARSING_TREE_BINARY_OP_MISSING_OPERANDS = 405, // During tree construction, a binary operator was found with fewer than two operands.
    PARSING_TREE_CHILD_STACK_OVERFLOW = 406,       // The stack for building child/parent pointers overflowed during tree construction.
    PARSING_TREE_MALFORMED_EXPRESSION = 407,       // Generic error for expressions that are syntactically incorrect (e.g., too many operands left on the stack after tree construction).

    // --- Postfix/Prefix-to-Infix Conversion Errors (from postfix_to_infix/prefix_to_infix) ---
    CONVERSION_ERROR_START = 500,
    CONVERSION_RESULTING_INFIX_TOO_LONG = 501,   // The resulting infix string exceeded the maximum allowed length.
    CONVERSION_INTERNAL_STACK_OVERFLOW = 502,    // Internal stack for building infix string overflowed.
    CONVERSION_UNARY_OP_MISSING_OPERAND = 503,   // Malformed input: a unary operator was found without an operand on the stack.
    CONVERSION_BINARY_OP_MISSING_OPERANDS = 504, // Malformed input: a binary operator was found with fewer than two operands on the stack.
    CONVERSION_MALFORMED_EXPRESSION = 505,       // Malformed input: e.g., too many items left on the stack at the end of conversion.

};

/**
 * @brief Converts an ErrorCode enum to a human-readable string.
 *
 * @param code The ErrorCode to convert.
 * @param context An optional string providing context for the error (e.g., "prefix parsing").
 * @return A descriptive std::string for the error.
 */
inline std::string getErrorMessage(ErrorCode code, const std::string &context = "")
{
    std::string base_message;

    switch (code)
    {
    // --- General ---
    case ErrorCode::NO_ERROR:
        base_message = "No error.";
        break;

    // --- Input Validation ---
    case ErrorCode::VALIDATION_CHILD_INDEX_GEQ_PARENT:
        base_message = "Validation failed: A child's index is greater than or equal to its parent's.";
        break;
    case ErrorCode::VALIDATION_ARITY_MISMATCH:
        base_message = "Validation failed: Operator arity does not match the number of children.";
        break;

    // --- Symbolic Evaluation Backward Pass ---
    case ErrorCode::EVAL_BACKWARD_GRAD_ON_NO_OP:
        base_message = "Backward pass error: A non-zero gradient was passed to a NO_OP node.";
        break;
    // Not used in Pytorch, so commented out.
    /*case ErrorCode::EVAL_BACKWARD_LOG_AT_NON_POSITIVE:
        base_message = "Backward pass error: Gradient for log(x) where x <= 0.";
        break;*/
    // Not used in Pytorch, so commented out.
    /*case ErrorCode::EVAL_BACKWARD_SQRT_AT_NEGATIVE:
        base_message = "Backward pass error: Gradient for sqrt(x) where x < 0.";
        break;*/
    case ErrorCode::EVAL_BACKWARD_DIV_BY_ZERO:
        base_message = "Backward pass error: Gradient for a/b where b = 0.";
        break;
    // Not used in Pytorch, so commented out.
    /*case ErrorCode::EVAL_BACKWARD_POW_WITH_NEGATIVE_BASE:
        base_message = "Backward pass error: Gradient for pow(a, b) where a < 0.";
        break;*/

    // --- PCFG Generation/Sampling ---
    case ErrorCode::GENERATION_STACK_OVERFLOW:
        base_message = "Generation failed: Internal stack exceeded capacity.";
        break;
    case ErrorCode::GENERATION_EXPRESSION_TOO_LONG:
        base_message = "Generation failed: Produced expression exceeded maximum length.";
        break;
    case ErrorCode::GENERATION_MAX_TRIES_EXCEEDED:
        base_message = "Generation failed: Could not produce a valid expression within the max_tries limit.";
        break;
    case ErrorCode::GENERATION_NO_VALID_RULE_FOUND:
        base_message = "Generation failed: No valid production rule found during sampling.";
        break;

    // --- Parsing ---
    case ErrorCode::PARSING_OUTPUT_QUEUE_OVERFLOW:
        base_message = "Parsing failed: Output queue overflowed (expression likely too complex).";
        break;
    case ErrorCode::PARSING_OPERATOR_STACK_OVERFLOW:
        base_message = "Parsing failed: Operator stack overflowed (expression likely too complex).";
        break;
    case ErrorCode::PARSING_MISMATCHED_PARENTHESIS:
        base_message = "Parsing failed: Mismatched or unbalanced parentheses.";
        break;
    case ErrorCode::PARSING_TREE_UNARY_OP_MISSING_OPERAND:
        base_message = "Parsing failed: Unary operator is missing an operand during tree construction.";
        break;
    case ErrorCode::PARSING_TREE_BINARY_OP_MISSING_OPERANDS:
        base_message = "Parsing failed: Binary operator is missing one or both operands during tree construction.";
        break;
    case ErrorCode::PARSING_TREE_CHILD_STACK_OVERFLOW:
        base_message = "Parsing failed: Child stack overflowed during tree construction.";
        break;
    case ErrorCode::PARSING_TREE_MALFORMED_EXPRESSION:
        base_message = "Parsing failed: Malformed expression (e.g., too many operands).";
        break;

    // --- Conversion ---
    case ErrorCode::CONVERSION_RESULTING_INFIX_TOO_LONG:
        base_message = "Conversion failed: Resulting infix expression is longer than the maximum allowed length.";
        break;
    case ErrorCode::CONVERSION_INTERNAL_STACK_OVERFLOW:
        base_message = "Conversion failed: Internal workspace stack overflowed.";
        break;
    case ErrorCode::CONVERSION_UNARY_OP_MISSING_OPERAND:
        base_message = "Conversion failed: Malformed input, unary operator is missing an operand.";
        break;
    case ErrorCode::CONVERSION_BINARY_OP_MISSING_OPERANDS:
        base_message = "Conversion failed: Malformed input, binary operator is missing one or both operands.";
        break;
    case ErrorCode::CONVERSION_MALFORMED_EXPRESSION:
        base_message = "Conversion failed: Malformed input (e.g., too many operands).";
        break;

    // --- Default ---
    default:
        std::stringstream ss;
        ss << "Unknown error code: " << static_cast<int64_t>(code);
        base_message = ss.str();
        break;
    }

    if (!context.empty())
    {
        return "Error during " + context + ": " + base_message;
    }

    return base_message;
}
