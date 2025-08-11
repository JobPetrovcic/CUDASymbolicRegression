#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include "operators.h"

/*
This file describes the data structures and algorithms for a batched PCFG expression generator in CUDA.

Each symbol in the grammar is given a unique integer ID. Non-terminals and terminals are distinguished.

This representation is efficient for GPU processing. We use pointer/offset arrays instead of start/end pairs.
- `rule_lhs`: An int tensor of shape (num_rules,). Stores the non-terminal ID (Left-Hand Side) for each rule.
- `rhs_ptr`: An int tensor of shape (num_rules + 1,). This is the rule-to-RHS pointer. The symbols for rule `i` are in `rhs_concat` from `rhs_ptr[i]` to `rhs_ptr[i+1] - 1`.
- `rhs_concat`: An int tensor of shape (total_rhs_symbols,). A concatenation of all Right-Hand Side symbols for all rules, in their natural order.
- `nt_rule_ptr`: An int tensor of shape (num_symbols + 1,). This is the symbol-to-rule pointer. The rules for `i` are in the probability/rule arrays from `nt_rule_ptr[i]` to `nt_rule_ptr[i+1] - 1`. If an ID is not a non-terminal, it will have no rules.
- `nt_rule_cum_probs`: A float tensor storing the CUMULATIVE probabilities for each non-terminal's rules, matching the order defined by `nt_rule_ptr`. This allows for fast sampling via binary search or a simple linear scan.


We also create a slower C++ version of this generation that uses multithreading for validation and debugging.
*/

constexpr float DEFAULT_tolerence = 1e-6f;
class ProbabilisticContextFreeGrammar
{
public:
    /*
    The constructor takes a string representation of the grammar and converts it into the internal CSR-style tensor representation described above.
    - It parses the grammar string rule by rule.
    - It builds a symbol-to-ID map.
    - It populates the `rule_lhs`, `rule_op_id`, `rhs_concat`, and probability vectors.
    - It sorts rules by non-terminal to group them.
    - It computes the final pointer (`_ptr`) and cumulative probability (`_cum_probs`) tensors.
    */
    ProbabilisticContextFreeGrammar(std::string grammar, std::string start_symbol, int64_t padded_maximum_length, int64_t n_variables, torch::Device device, int64_t max_tries = 100, float tolerance = DEFAULT_tolerence, bool verbose = false);
    ~ProbabilisticContextFreeGrammar();
    torch::Tensor sample_infix(int64_t B);
    std::vector<std::string> to_string(torch::Tensor expressions);

    torch::Tensor infix_to_prefix(torch::Tensor expressions, int verbosity = 0);
    torch::Tensor infix_to_postfix(torch::Tensor expressions, int verbosity = 0);

    torch::Tensor postfix_to_infix(torch::Tensor expressions, int64_t max_infix_len, int verbosity = 0);
    torch::Tensor prefix_to_infix(torch::Tensor expressions, int64_t max_infix_len, int verbosity = 0);
    torch::Tensor prefix_to_postfix(torch::Tensor expressions, int verbosity = 0);
    torch::Tensor postfix_to_prefix(torch::Tensor expressions, int verbosity = 0);

    torch::Tensor get_prefix_parent(torch::Tensor expressions, int verbosity = 0);
    torch::Tensor get_postfix_children(torch::Tensor expressions, int verbosity = 0);

    torch::Device device;

    // --- Exposed Parameters ---
    std::string start_symbol;
    int64_t padded_maximum_length;
    int64_t max_tries;
    int64_t n_variables;
    float tolerance;
    bool verbose;

    // Getters symbol
    int64_t get_symbol_id(const std::string &symbol) const
    {
        auto it = this->symbol_to_id.find(symbol);
        if (it != this->symbol_to_id.end())
        {
            return it->second;
        }
        throw std::runtime_error("Symbol not found: " + symbol);
    }
    int64_t n_operators;
    torch::Tensor get_arities() const
    {
        torch::Tensor arities = torch::zeros({this->n_operators}, torch::kInt64).to(device);
        for (int64_t i = 0; i < this->n_operators; ++i)
        {
            arities[i] = get_arity(i);
        }
        return arities;
    }
    torch::Tensor valid_ops() const
    {
        torch::Tensor valid_ops = torch::zeros({this->n_operators}, torch::kBool).to(device);
        for (int64_t i = 0; i < this->n_operators; ++i)
        {
            valid_ops[i] = is_valid_op(i, this->n_variables);
        }
        return valid_ops;
    }
    std::vector<std::string> available_operators() const;

private:
    int64_t terminal_limit;
    int64_t new_symbol_id_t;

    std::unordered_map<std::string, int64_t> symbol_to_id;
    std::unordered_map<int64_t, std::string> id_to_symbol;

    void add_symbol(const std::string &s, int64_t id);
    void get_initial_symbol_map_and_precedence(int64_t n_variables);
    int64_t get_token_id(std::string s);
    std::vector<int64_t> parse_sides(std::string s, std::string line);

    void process_parsing_errors(const torch::Tensor &errors, const torch::Tensor &expressions, const std::string &context, int verbosity = 0);

    // Checks
    void infix_checks(torch::Tensor infix_ops);
    void prefix_checks(torch::Tensor prefix_ops);
    void postfix_checks(torch::Tensor postfix_ops);

    torch::Tensor id_to_pos;
    torch::Tensor rule_lhs;          // Shape: (num_rules,)
    torch::Tensor rhs_ptr;           // Shape: (num_rules + 1,)
    torch::Tensor rhs_concat;        // Shape: (total_rhs_symbols,)
    torch::Tensor nt_rule_ptr;       // Shape: (num_symbols + 1,)
    torch::Tensor nt_rule_cum_probs; // Shape: (num_rules,)
    torch::Tensor precedence;        // Shape: (num_symbols,)
};