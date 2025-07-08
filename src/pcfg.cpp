#include "pcfg.h"
#include "pcfg_kernels.h"
#include "operators.h"
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Helper function to trim leading/trailing whitespace from a string
std::string trim(const std::string &s)
{
    auto start = s.begin();
    while (start != s.end() && std::isspace(static_cast<unsigned char>(*start)))
    {
        start++;
    }

    if (start == s.end())
    {
        return "";
    }

    auto end = s.end();
    do
    {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));

    return std::string(start, end + 1);
}

/*

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
*/

void ProbababilisticContextFreeGrammar::add_symbol(const std::string &s, int64_t id)
{
    symbol_to_id[s] = id;
    if (id_to_symbol.find(id) == id_to_symbol.end())
    {
        id_to_symbol[id] = s;
    }
}

// TODO: check terminal bounds
void ProbababilisticContextFreeGrammar::get_initial_symbol_map_and_precedence(int64_t n_variables)
{
    symbol_to_id = std::unordered_map<std::string, int64_t>();
    id_to_symbol = std::unordered_map<int64_t, std::string>();
    add_symbol("C", LEARNABLE_CONSTANT);

    add_symbol("1", CONST_1);

    add_symbol("sin", SIN);
    add_symbol("SIN", SIN);
    add_symbol("Sin", SIN);

    add_symbol("cos", COS);
    add_symbol("COS", COS);
    add_symbol("Cos", COS);

    add_symbol("exp", EXP);
    add_symbol("EXP", EXP);
    add_symbol("Exp", EXP);

    add_symbol("log", LOG);
    add_symbol("LOG", LOG);
    add_symbol("Log", LOG);
    add_symbol("ln", LOG);
    add_symbol("LN", LOG);
    add_symbol("Ln", LOG);

    add_symbol("^2", SQUARE);
    add_symbol("square", SQUARE);
    add_symbol("SQUARE", SQUARE);
    add_symbol("Square", SQUARE);

    add_symbol("sqrt", SQRT);
    add_symbol("SQRT", SQRT);
    add_symbol("Sqrt", SQRT);

    add_symbol("+", ADD);

    add_symbol("-", SUB);

    add_symbol("*", MUL);

    add_symbol("/", DIV);

    for (int64_t i = 0; i < n_variables; ++i)
    {
        std::string var_name = "X_" + std::to_string(i);
        add_symbol(var_name, VAR_START_ID + i);
    }

    add_symbol("(", VAR_START_ID + n_variables);
    add_symbol(")", VAR_START_ID + n_variables + 1);

    // END OF TERMINALS
    new_symbol_id_t = VAR_START_ID + n_variables + 2;
    // Init precedence tensor
    this->precedence = torch::zeros({new_symbol_id_t}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    // Constants and Variables have lowest precedence
    this->precedence.index_put_({LEARNABLE_CONSTANT}, 0); // C
    this->precedence.index_put_({CONST_1}, 0);            // 1
    for (int64_t i = 0; i < n_variables; ++i)
    {
        this->precedence.index_put_({VAR_START_ID + i}, 0); // X_i
    }
    // Level 2: Binary function +, -, *, /
    this->precedence.index_put_({ADD}, 1); // +
    this->precedence.index_put_({SUB}, 1); // -
    this->precedence.index_put_({MUL}, 2); // *
    this->precedence.index_put_({DIV}, 2); // /
    // Level 3: Unary functions
    this->precedence.index_put_({SIN}, 3);    // sin
    this->precedence.index_put_({COS}, 3);    // cos
    this->precedence.index_put_({EXP}, 3);    // exp
    this->precedence.index_put_({LOG}, 3);    // log
    this->precedence.index_put_({SQUARE}, 3); // square
    this->precedence.index_put_({SQRT}, 3);   // sqrt

    this->terminal_limit = VAR_START_ID + n_variables;

    // Level 4: Parentheses
    this->precedence.index_put_({VAR_START_ID + n_variables}, 4);     // (
    this->precedence.index_put_({VAR_START_ID + n_variables + 1}, 4); // )

    // START OF NON-TERMINALS
}

int64_t ProbababilisticContextFreeGrammar::get_token_id(std::string s)
{
    auto it = symbol_to_id.find(s);
    if (it != symbol_to_id.end())
    {
        return it->second;
    }
    else
    {
        // If the symbol is not found, assign a new ID
        int64_t new_id = new_symbol_id_t++;
        symbol_to_id[s] = new_id;
        return new_id;
    }
}

std::vector<int64_t> ProbababilisticContextFreeGrammar::parse_sides(std::string s)
{
    std::vector<int64_t> result;
    std::istringstream ss(s);
    std::string token;
    while (ss >> token)
    {
        int64_t token_id = get_token_id(token);
        result.push_back(token_id);
    }
    return result;
}

ProbababilisticContextFreeGrammar::ProbababilisticContextFreeGrammar(std::string grammar, std::string start_symbol, int64_t padded_maximum_length, int64_t n_variables, torch::Device device, int64_t max_tries, float tolerance)
    : start_symbol(std::move(start_symbol)),
      padded_maximum_length(padded_maximum_length),
      device(device),
      tolerance(tolerance),
      max_tries(max_tries),
      n_variables(n_variables)
{
    if (padded_maximum_length > HARD_MAX_LENGTH)
    {
        throw std::invalid_argument("padded_maximum_length must be less than or equal to HARD_MAX_LENGTH (" + std::to_string(HARD_MAX_LENGTH) + ")");
    }
    get_initial_symbol_map_and_precedence(n_variables);

    // Loop through lines: one rule <-> one line
    // Example: E -> E '+' F [0.2]
    std::istringstream grammar_stream(grammar);
    std::string line;

    std::vector<int64_t> rule_lhs_vec;
    std::vector<std::vector<int64_t>> rhsides;
    std::vector<float> probabilities;

    while (std::getline(grammar_stream, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        // Split the line into three parts: LHS, RHS, and probability
        size_t arrow_pos = line.find("->");
        if (arrow_pos == std::string::npos)
            throw std::invalid_argument("Invalid grammar rule: " + line);

        std::string lhs = trim(line.substr(0, arrow_pos));
        std::string rhs_and_prob = trim(line.substr(arrow_pos + 2));
        // throw if multiple '->' in the line
        if (rhs_and_prob.find("->") != std::string::npos)
        {
            throw std::invalid_argument("Invalid grammar rule with multiple '->': " + line);
        }

        // Split RHS and probability
        size_t prob_pos = rhs_and_prob.find('[');
        if (prob_pos == std::string::npos)
            throw std::invalid_argument("Invalid grammar rule without probability: " + line);

        std::string rhs = trim(rhs_and_prob.substr(0, prob_pos));
        std::string prob_str = rhs_and_prob.substr(prob_pos + 1);
        if (prob_str.back() != ']')
        {
            throw std::invalid_argument("Invalid grammar rule with missing ']': " + line);
        }

        // If there are multiple ']' in the line, throw an error
        if (std::count(prob_str.begin(), prob_str.end(), ']') > 1)
        {
            throw std::invalid_argument("Invalid grammar rule with multiple ']' characters: " + line);
        }
        prob_str.pop_back(); // Remove the closing bracket
        float probability = std::stof(trim(prob_str));

        // Parse LHS and RHS
        std::vector<int64_t> lhs_tokens = parse_sides(lhs);
        if (lhs_tokens.size() != 1)
        {
            throw std::invalid_argument("LHS must contain exactly one non-terminal: " + lhs);
        }
        int64_t lhs_token_id = lhs_tokens[0];
        std::vector<int64_t> rhs_tokens = parse_sides(rhs);

        if (rhs_tokens.empty())
        {
            throw std::invalid_argument("RHS cannot be empty: " + rhs);
        }

        // Store the rule
        rule_lhs_vec.push_back(lhs_token_id);
        rhsides.push_back(rhs_tokens);
        probabilities.push_back(probability);
    }

    // Now we have all rules parsed, we need to build the CSR-style representation
    // Step 1: sort the rules by LHS
    std::vector<int64_t> sorted_indices(rule_lhs_vec.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&rule_lhs_vec](int64_t a, int64_t b)
              {
                  return rule_lhs_vec[a] < rule_lhs_vec[b];
              });
    // Permute the rules according to sorted indices
    std::vector<int64_t> sorted_rule_lhs;
    std::vector<std::vector<int64_t>> sorted_rhsides;
    std::vector<float> sorted_probabilities;
    for (int64_t idx : sorted_indices)
    {
        sorted_rule_lhs.push_back(rule_lhs_vec[idx]);
        sorted_rhsides.push_back(rhsides[idx]);
        sorted_probabilities.push_back(probabilities[idx]);
    }

    // Now we have all rules parsed, we need to build the CSR-style representation
    // Step 2: build the tensor representations

    // Build rhs_concat and rhs_ptr
    std::vector<int64_t> rhs_concat_vec;
    std::vector<int64_t> rhs_ptr_vec;
    rhs_ptr_vec.push_back(0);
    for (const auto &rhs : sorted_rhsides)
    {
        rhs_concat_vec.insert(rhs_concat_vec.end(), rhs.begin(), rhs.end());
        rhs_ptr_vec.push_back(rhs_concat_vec.size());
    }
    this->rhs_concat = torch::tensor(rhs_concat_vec, torch::kInt64);
    this->rhs_ptr = torch::tensor(rhs_ptr_vec, torch::kInt64);

    // Build rule_lhs
    this->rule_lhs = torch::tensor(sorted_rule_lhs, torch::kInt64);

    if (sorted_rule_lhs.empty())
    {
        throw std::invalid_argument("No rules found in the grammar.");
    }
    // Build nt_rule_ptr and nt_rule_cum_probs
    std::vector<int64_t> nt_rule_ptr_vec(new_symbol_id_t + 1, 0);
    nt_rule_ptr_vec[sorted_rule_lhs[0]] = 0;
    for (size_t i = 1; i < sorted_rule_lhs.size(); ++i)
    {
        for (int64_t j = sorted_rule_lhs[i - 1] + 1; j <= sorted_rule_lhs[i]; ++j)
        {
            nt_rule_ptr_vec[j] = i;
        }
    }
    for (int64_t j = sorted_rule_lhs.back() + 1; j <= new_symbol_id_t; ++j)
    {
        nt_rule_ptr_vec[j] = sorted_rule_lhs.size();
    }

    std::vector<float> nt_rule_cum_probs_vec(sorted_rule_lhs.size());
    for (int64_t i = 0; i < new_symbol_id_t; ++i)
    {
        int64_t start_idx = nt_rule_ptr_vec[i];
        int64_t end_idx = nt_rule_ptr_vec[i + 1];
        if (start_idx == end_idx)
            continue;

        double total_prob = 0.0;
        for (int64_t j = start_idx; j < end_idx; ++j)
        {
            total_prob += sorted_probabilities[j];
        }

        if (std::abs(total_prob - 1.0) > this->tolerance)
        {
            auto nonterminal_symbol = id_to_symbol[i];
            throw std::invalid_argument("Non-terminal '" + nonterminal_symbol + "' does not sum to 1.0. Total probability: " + std::to_string(total_prob) + ".");
        }

        double cum_prob = 0.0;
        for (int64_t j = start_idx; j < end_idx; ++j)
        {
            cum_prob += sorted_probabilities[j];
            nt_rule_cum_probs_vec[j] = cum_prob;
        }
        if (end_idx > start_idx)
        {
            nt_rule_cum_probs_vec[end_idx - 1] = 1.0; // Ensure it sums to 1
        }
    }
    this->nt_rule_ptr = torch::tensor(nt_rule_ptr_vec, torch::kInt64);
    this->nt_rule_cum_probs = torch::tensor(nt_rule_cum_probs_vec, torch::kFloat32);

    // Move tensors to the specified device
    this->rule_lhs = this->rule_lhs.to(device);
    this->rhs_ptr = this->rhs_ptr.to(device);
    this->rhs_concat = this->rhs_concat.to(device);
    this->nt_rule_ptr = this->nt_rule_ptr.to(device);
    this->nt_rule_cum_probs = this->nt_rule_cum_probs.to(device);

    // Check that if a symbol has no rules, it is a terminal (id < terminal_limit)
    for (int64_t i = terminal_limit; i < new_symbol_id_t; ++i)
    {
        int64_t start_idx = nt_rule_ptr[i].item<int64_t>();
        int64_t end_idx = nt_rule_ptr[i + 1].item<int64_t>();
        if (start_idx == end_idx)
        {
            // This symbol has no rules, it must be a terminal
            if (id_to_symbol.find(i) == id_to_symbol.end())
            {
                throw std::invalid_argument("Symbol with ID " + std::to_string(i) + " has no rules and is not a terminal.");
            }
        }
    }
}

ProbababilisticContextFreeGrammar::~ProbababilisticContextFreeGrammar() {}

torch::Tensor ProbababilisticContextFreeGrammar::sample_string_expression(int64_t B)
{
    auto output = torch::empty({B, padded_maximum_length}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto errors = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    int64_t start_symbol_id = get_token_id(start_symbol);

    // Init seeds for random number generation
    auto seeds = torch::randint(0, INT64_MAX, {B}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    if (device.is_cuda())
    {
        pcfg_sample_string_expression_cuda_impl(
            this->rule_lhs.packed_accessor32<int64_t, 1>(),
            this->rhs_ptr.packed_accessor32<int64_t, 1>(),
            this->rhs_concat.packed_accessor32<int64_t, 1>(),
            this->nt_rule_ptr.packed_accessor32<int64_t, 1>(),
            this->nt_rule_cum_probs.packed_accessor32<float, 1>(),
            start_symbol_id,
            padded_maximum_length,
            max_tries, // max_tries
            output.packed_accessor32<int64_t, 2>(),
            errors.packed_accessor32<int64_t, 1>(),
            seeds.packed_accessor32<int64_t, 1>(),
            B);
    }
    else
    {
        pcfg_sample_string_expression_cpu_impl(
            this->rule_lhs.accessor<int64_t, 1>(),
            this->rhs_ptr.accessor<int64_t, 1>(),
            this->rhs_concat.accessor<int64_t, 1>(),
            this->nt_rule_ptr.accessor<int64_t, 1>(),
            this->nt_rule_cum_probs.accessor<float, 1>(),
            start_symbol_id,
            padded_maximum_length,
            max_tries, // max_tries
            output.accessor<int64_t, 2>(),
            errors.accessor<int64_t, 1>(),
            seeds.accessor<int64_t, 1>(),
            B);
    }
    if (errors.any().item<bool>())
    {
        std::stringstream ss;
        ss << "Failed to generate expressions for some batch items. Error counts:\n";
        auto errors_cpu = errors.to(torch::kCPU);
        auto errors_acc = errors_cpu.accessor<int64_t, 1>();
        std::map<int64_t, int64_t> error_counts;
        for (int64_t i = 0; i < errors_acc.size(0); ++i)
        {
            if (errors_acc[i] != 0)
                error_counts[errors_acc[i]]++;
        }

        for (const auto &pair : error_counts)
        {
            ss << "  Error " << pair.first << ": " << pair.second << " times. ";
            switch (pair.first)
            {
            case 1:
                ss << "(Stack overflow or no valid rule found)";
                break;
            case 9:
                ss << "(Failed to generate expression after max_tries)";
                break;
            case 10:
                ss << "(No valid rule found)";
                break;
            case 11:
                ss << "(Stack overflow)";
                break;
            default:
                ss << "(Unknown error code)";
                break;
            }
            ss << "\n";
        }
        throw std::runtime_error(ss.str());
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> ProbababilisticContextFreeGrammar::parse_to_postfix(torch::Tensor expressions)
{
    // Check that only terminal symbols are used
    TORCH_CHECK((0 <= expressions).all().item<bool>(),
                "All symbols in the expressions must be non-negative integers.");
    TORCH_CHECK((expressions < this->terminal_limit).item<bool>(),
                "All symbols in the expressions must be terminal symbols (less than terminal_limit).");

    int64_t B = expressions.size(0);
    int64_t M = expressions.size(1);

    if (M > HARD_MAX_LENGTH)
    {
        throw std::invalid_argument("M must be less than or equal to HARD_MAX_LENGTH, but got " + std::to_string(M) + " > " + std::to_string(HARD_MAX_LENGTH) + ".");
    }

    auto ops = torch::full({B, M}, NO_OP, expressions.options());
    auto children = torch::full({B, M, 2}, NULL_CHILD, expressions.options());
    auto errors = torch::zeros({B}, expressions.options().dtype(torch::kInt64));

    int64_t lparenthesis_id = get_token_id("(");
    int64_t rparenthesis_id = get_token_id(")");

    if (device.is_cuda())
    {
        parse_to_postfix_cuda_impl(
            this->precedence.packed_accessor32<int64_t, 1>(),
            expressions.packed_accessor32<int64_t, 2>(),
            ops.packed_accessor32<int64_t, 2>(),
            children.packed_accessor32<int64_t, 3>(),
            errors.packed_accessor32<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }
    else
    {
        parse_to_postfix_cpu_impl(
            this->precedence.accessor<int64_t, 1>(),
            expressions.accessor<int64_t, 2>(),
            ops.accessor<int64_t, 2>(),
            children.accessor<int64_t, 3>(),
            errors.accessor<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }

    if (errors.any().item<bool>())
    {
        std::stringstream ss;
        ss << "Failed to parse expressions for some batch items. Error counts:\n";
        auto errors_cpu = errors.to(torch::kCPU);
        auto errors_acc = errors_cpu.accessor<int64_t, 1>();
        std::map<int64_t, int64_t> error_counts;
        for (int64_t i = 0; i < errors_acc.size(0); ++i)
        {
            if (errors_acc[i] != 0)
                error_counts[errors_acc[i]]++;
        }

        for (const auto &pair : error_counts)
        {
            ss << "  Error " << pair.first << ": " << pair.second << " times. ";
            switch (pair.first)
            {
            case 1:
                ss << "(Postfix expression too long)";
                break;
            case 2:
                ss << "(Operator stack overflow)";
                break;
            case 3:
                ss << "(Mismatched parenthesis)";
                break;
            case 5:
                ss << "(Unary operator without operand)";
                break;
            case 6:
                ss << "(Binary operator without enough operands)";
                break;
            case 7:
                ss << "(Child stack overflow)";
                break;
            case 8:
                ss << "(Malformed expression)";
                break;
            case 9:
                ss << "(Failed to generate expression after max_tries)";
                break;
            default:
                ss << "(Unknown error code)";
                break;
            }
            ss << "\n";
        }
        throw std::runtime_error(ss.str());
    }

    return std::make_tuple(ops, children);
}

std::tuple<torch::Tensor, torch::Tensor> ProbababilisticContextFreeGrammar::sample(int64_t B)
{
    if (B <= 0)
        throw std::invalid_argument("Batch size B must be greater than 0, but got " + std::to_string(B) + ".");

    auto string_samples = sample_string_expression(B);
    return parse_to_postfix(string_samples);
}

std::vector<std::string> ProbababilisticContextFreeGrammar::to_string(torch::Tensor expressions)
{
    expressions = expressions.to(torch::kCPU);
    auto accessor = expressions.accessor<int64_t, 2>();
    int64_t B = accessor.size(0);
    int64_t M = accessor.size(1);
    std::vector<std::string> results;
    results.reserve(B);

    for (int64_t i = 0; i < B; ++i)
    {
        std::stringstream ss;
        for (int64_t j = 0; j < M; ++j)
        {
            int64_t token_id = accessor[i][j];
            if (token_id == 0) // NO_OP is padding TODO: use casting
                break;
            auto it = id_to_symbol.find(token_id);
            if (it != id_to_symbol.end())
            {
                ss << it->second << " ";
            }
        }
        std::string s = ss.str();
        if (!s.empty())
        {
            s.pop_back(); // remove last space
        }
        results.push_back(s);
    }
    return results;
}

// Export using PyBind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void init_pcfg(pybind11::module &m)
{
    pybind11::class_<ProbababilisticContextFreeGrammar>(m, "ProbababilisticContextFreeGrammar")
        .def(pybind11::init<std::string, std::string, int64_t, int64_t, torch::Device, int64_t, float>(),
             pybind11::arg("grammar"),
             pybind11::arg("start_symbol"),
             pybind11::arg("padded_maximum_length"),
             pybind11::arg("n_variables"),
             pybind11::arg("device"),
             pybind11::arg("max_tries") = 100,
             pybind11::arg("tolerance") = DEFAULT_tolerence)
        .def("sample_string_expression", &ProbababilisticContextFreeGrammar::sample_string_expression, "Sample string expressions from the grammar")
        .def("sample", &ProbababilisticContextFreeGrammar::sample)
        .def("to_string", &ProbababilisticContextFreeGrammar::to_string)
        .def("parse_to_postfix", &ProbababilisticContextFreeGrammar::parse_to_postfix)
        .def("sample_string_expression", &ProbababilisticContextFreeGrammar::sample_string_expression);
}