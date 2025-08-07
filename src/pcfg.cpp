#include "pcfg.h"
#include "pcfg_kernels.h"
#include "operators.h"
#include "error_codes.h"
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

void ProbabilisticContextFreeGrammar::add_symbol(const std::string &s, int64_t id)
{
    symbol_to_id[s] = id;
    if (id_to_symbol.find(id) == id_to_symbol.end())
    {
        id_to_symbol[id] = s;
    }
}

// TODO: check terminal bounds
void ProbabilisticContextFreeGrammar::get_initial_symbol_map_and_precedence(int64_t n_variables)
{
    symbol_to_id = std::unordered_map<std::string, int64_t>();
    id_to_symbol = std::unordered_map<int64_t, std::string>();

    add_symbol("NO_OP", NO_OP); // TODO: this might cause some errors, run the tests
    add_symbol("C", LEARNABLE_CONSTANT);

    add_symbol("1", CONST_1);
    add_symbol("2", CONST_2);
    add_symbol("3", CONST_3);
    add_symbol("4", CONST_4);
    add_symbol("5", CONST_5);
    add_symbol("pi", PI);
    add_symbol("PI", PI);
    add_symbol("e", E);
    add_symbol("E", E);

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

    add_symbol("tan", TAN);
    add_symbol("TAN", TAN);
    add_symbol("Tan", TAN);

    add_symbol("arcsin", ARCSIN);
    add_symbol("ARCSIN", ARCSIN);
    add_symbol("asin", ARCSIN);
    add_symbol("ASIN", ARCSIN);

    add_symbol("arccos", ARCCOS);
    add_symbol("ARCCOS", ARCCOS);
    add_symbol("acos", ARCCOS);
    add_symbol("ACOS", ARCCOS);

    add_symbol("arctan", ARCTAN);
    add_symbol("ARCTAN", ARCTAN);
    add_symbol("atan", ARCTAN);
    add_symbol("ATAN", ARCTAN);

    add_symbol("sinh", SINH);
    add_symbol("SINH", SINH);
    add_symbol("Sinh", SINH);

    add_symbol("cosh", COSH);
    add_symbol("COSH", COSH);
    add_symbol("Cosh", COSH);

    add_symbol("tanh", TANH);
    add_symbol("TANH", TANH);
    add_symbol("Tanh", TANH);

    add_symbol("floor", FLOOR);
    add_symbol("FLOOR", FLOOR);
    add_symbol("Floor", FLOOR);

    add_symbol("ceil", CEIL);
    add_symbol("CEIL", CEIL);
    add_symbol("Ceil", CEIL);

    add_symbol("ln", LN);
    add_symbol("LN", LN);

    add_symbol("log10", LOG10);
    add_symbol("LOG10", LOG10);

    add_symbol("neg", NEG);
    add_symbol("NEG", NEG);
    add_symbol("minus", NEG);
    add_symbol("-u", NEG);

    add_symbol("inv", INV);
    add_symbol("INV", INV);
    add_symbol("^-1", INV);

    add_symbol("cube", CUBE);
    add_symbol("CUBE", CUBE);
    add_symbol("^3", CUBE);

    add_symbol("fourth", FOURTH);
    add_symbol("FOURTH", FOURTH);
    add_symbol("^4", FOURTH);

    add_symbol("fifth", FIFTH);
    add_symbol("FIFTH", FIFTH);
    add_symbol("^5", FIFTH);

    add_symbol("+", ADD);

    add_symbol("-", SUB);

    add_symbol("*", MUL);

    add_symbol("/", DIV);

    add_symbol("^", POW);

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
    this->precedence.index_put_({CONST_2}, 0);
    this->precedence.index_put_({CONST_3}, 0);
    this->precedence.index_put_({CONST_4}, 0);
    this->precedence.index_put_({CONST_5}, 0);
    this->precedence.index_put_({PI}, 0);
    this->precedence.index_put_({E}, 0);
    for (int64_t i = 0; i < n_variables; ++i)
    {
        this->precedence.index_put_({VAR_START_ID + i}, 0); // X_i
    }
    // Level 1: ADD, SUB
    this->precedence.index_put_({ADD}, 1); // +
    this->precedence.index_put_({SUB}, 1); // -
    // Level 2: MUL, DIV
    this->precedence.index_put_({MUL}, 2); // *
    this->precedence.index_put_({DIV}, 2); // /
    // Level 3: POW
    this->precedence.index_put_({POW}, 3); // ^
    // Level 4: Unary functions
    this->precedence.index_put_({SIN}, 4);
    this->precedence.index_put_({COS}, 4);
    this->precedence.index_put_({EXP}, 4);
    this->precedence.index_put_({LOG}, 4);
    this->precedence.index_put_({SQUARE}, 4);
    this->precedence.index_put_({SQRT}, 4);
    this->precedence.index_put_({TAN}, 4);
    this->precedence.index_put_({ARCSIN}, 4);
    this->precedence.index_put_({ARCCOS}, 4);
    this->precedence.index_put_({ARCTAN}, 4);
    this->precedence.index_put_({SINH}, 4);
    this->precedence.index_put_({COSH}, 4);
    this->precedence.index_put_({TANH}, 4);
    this->precedence.index_put_({FLOOR}, 4);
    this->precedence.index_put_({CEIL}, 4);
    this->precedence.index_put_({LN}, 4);
    this->precedence.index_put_({LOG10}, 4);
    this->precedence.index_put_({NEG}, 4);
    this->precedence.index_put_({INV}, 4);
    this->precedence.index_put_({CUBE}, 4);
    this->precedence.index_put_({FOURTH}, 4);
    this->precedence.index_put_({FIFTH}, 4);

    this->n_operators = VAR_START_ID + n_variables;

    this->terminal_limit = VAR_START_ID + n_variables + 2;

    // Level 4: Parentheses
    this->precedence.index_put_({VAR_START_ID + n_variables}, 4);     // (
    this->precedence.index_put_({VAR_START_ID + n_variables + 1}, 4); // )

    // START OF NON-TERMINALS
}

int64_t ProbabilisticContextFreeGrammar::get_token_id(std::string s)
{
    if (s.empty())
    {
        throw std::invalid_argument("Symbol cannot be empty.");
    }
    auto it = symbol_to_id.find(s);
    if (it != symbol_to_id.end())
    {
        return it->second;
    }
    else
    {
        // If the symbol is not found, assign a new ID
        int64_t new_id = new_symbol_id_t++;
        add_symbol(s, new_id);
        return new_id;
    }
}

std::vector<int64_t> ProbabilisticContextFreeGrammar::parse_sides(std::string s, std::string line)
{
    std::vector<int64_t> result;
    std::istringstream ss(s);
    std::string token;
    while (ss >> token)
    {
        if (token.empty())
            throw std::invalid_argument("Empty token found in right hand side grammar rule: " + line);
        int64_t token_id = get_token_id(token);
        result.push_back(token_id);
    }
    return result;
}

ProbabilisticContextFreeGrammar::ProbabilisticContextFreeGrammar(std::string grammar, std::string start_symbol_param, int64_t padded_maximum_length, int64_t n_variables, torch::Device device, int64_t max_tries, float tolerance, bool verbose)
    : start_symbol(std::move(start_symbol_param)),
      padded_maximum_length(padded_maximum_length),
      device(device),
      tolerance(tolerance),
      max_tries(max_tries),
      n_variables(n_variables),
      verbose(verbose)
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

        if (lhs.empty())
        {
            throw std::invalid_argument("LHS cannot be empty in rule: " + line);
        }
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
        std::vector<int64_t> lhs_tokens = parse_sides(lhs, line);
        if (lhs_tokens.size() != 1)
        {
            throw std::invalid_argument("LHS must contain exactly one non-terminal: " + lhs);
        }
        int64_t lhs_token_id = lhs_tokens[0];
        std::vector<int64_t> rhs_tokens = parse_sides(rhs, line);

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

            throw std::invalid_argument("Symbol " + id_to_symbol[i] + " with ID " + std::to_string(i) + " has no rules and is not a terminal.");
        }
    }

    if (verbose)
    {
        std::cout << "ProbabilisticContextFreeGrammar initialized with " << sorted_rule_lhs.size() << " rules." << std::endl;
        std::cout << "\tStart symbol: " << this->start_symbol;
        std::cout << ", ID: " << get_token_id(this->start_symbol) << std::endl;
        std::cout << "\tMaximum length: " << padded_maximum_length << std::endl;
        std::cout << "\tNumber of variables: " << n_variables << std::endl;
        std::cout << "\tDevice: " << device << std::endl;
        std::cout << "\tMax tries: " << max_tries << std::endl;
        std::cout << "\tTolerance: " << tolerance << std::endl;
        for (const auto &pair : symbol_to_id)
        {
            std::cout << "\tSymbol: " << pair.first << ", ID: " << pair.second << std::endl;
        }

        // Print the processed grammar rules
        std::cout << "Processed grammar rules:" << std::endl;
        for (size_t i = 0; i < sorted_rule_lhs.size(); ++i)
        {
            std::cout << "\tRule " << i + 1 << ": " << id_to_symbol[sorted_rule_lhs[i]] << " -> ";
            for (const auto &rhs : sorted_rhsides[i])
            {
                std::cout << id_to_symbol[rhs] << " ";
            }
            std::cout << "[" << sorted_probabilities[i] << "]" << std::endl;
        }
    }
}

ProbabilisticContextFreeGrammar::~ProbabilisticContextFreeGrammar() {}

void ProbabilisticContextFreeGrammar::process_parsing_errors(const torch::Tensor &errors, const torch::Tensor &expressions, const std::string &context, int verbosity)
{
    if (!errors.any().item<bool>())
    {
        return; // No errors to process
    }

    // Move to CPU for easier processing
    auto errors_cpu = errors.to(torch::kCPU);
    auto errors_acc = errors_cpu.accessor<int64_t, 1>();

    // --- Level 0: Count errors and print summary ---
    std::stringstream ss;
    ss << "Failed to parse expressions during " << context << " for some batch items.\n";

    std::map<int64_t, int64_t> error_counts;
    for (int64_t i = 0; i < errors_acc.size(0); ++i)
    {
        if (errors_acc[i] != 0)
        {
            error_counts[errors_acc[i]]++;
        }
    }

    ss << "Error Summary:\n";
    for (const auto &pair : error_counts)
    {
        ErrorCode code = static_cast<ErrorCode>(pair.first);
        ss << "  " << getErrorMessage(code) << ": " << pair.second << " occurrences.\n";
    }

    // --- Levels 1 & 2: Print problematic expressions ---
    if (verbosity > 0)
    {
        auto error_indices = torch::nonzero(errors_cpu).squeeze(-1);
        int64_t num_errors = error_indices.size(0);
        int64_t limit = (verbosity == 1) ? std::min((int64_t)5, num_errors) : num_errors;

        ss << "\nDisplaying " << limit << " out of " << num_errors << " problematic expressions:\n";

        auto expressions_cpu = expressions.to(torch::kCPU);

        for (int64_t i = 0; i < limit; ++i)
        {
            int64_t bad_idx = error_indices[i].item<int64_t>();
            int64_t error_code_val = errors_acc[bad_idx];
            ErrorCode code = static_cast<ErrorCode>(error_code_val);

            // Get the single expression tensor and convert it to string
            auto single_expr_tensor = expressions_cpu.index({bad_idx}).unsqueeze(0);
            std::vector<std::string> expr_str_vec = this->to_string(single_expr_tensor);

            ss << "----------------------------------------\n";
            ss << "Problem at Index: " << bad_idx << "\n";
            ss << "  Error: " << getErrorMessage(code) << "\n";
            ss << "  Input Expression: '" << expr_str_vec[0] << "'\n";
            ss << "  Expression Tensor: " << single_expr_tensor << "\n";
        }
        if (num_errors > limit)
        {
            ss << "----------------------------------------\n";
            ss << "... and " << (num_errors - limit) << " more.\n";
        }
    }

    throw std::runtime_error(ss.str());
}

torch::Tensor ProbabilisticContextFreeGrammar::sample_string_expression(int64_t B)
{
    auto output = torch::empty({B, padded_maximum_length}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto errors = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    int64_t start_symbol_id = get_token_id(this->start_symbol);

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
            ErrorCode code = static_cast<ErrorCode>(pair.first);
            ss << "  " << getErrorMessage(code) << ": " << pair.second << " occurrences.\n";
        }
        throw std::runtime_error(ss.str());
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> ProbabilisticContextFreeGrammar::parse_to_prefix(torch::Tensor expressions, int verbosity)
{
    // Check that only terminal symbols are used
    TORCH_CHECK((0 <= expressions).all().item<bool>(),
                "All symbols in the expressions must be non-negative integers.");
    TORCH_CHECK((expressions < this->terminal_limit).all().item<bool>(),
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
        parse_to_prefix_cuda_impl(
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
        parse_to_prefix_cpu_impl(
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

    process_parsing_errors(errors, expressions, "prefix parsing", verbosity);

    return std::make_tuple(ops, children);
}

std::tuple<torch::Tensor, torch::Tensor> ProbabilisticContextFreeGrammar::parse_to_prefix_parent(torch::Tensor expressions, int verbosity)
{
    // Check that only terminal symbols are used
    TORCH_CHECK((0 <= expressions).all().item<bool>(),
                "All symbols in the expressions must be non-negative integers.");
    TORCH_CHECK((expressions < this->terminal_limit).all().item<bool>(),
                "All symbols in the expressions must be terminal symbols (less than terminal_limit).");

    int64_t B = expressions.size(0);
    int64_t M = expressions.size(1);

    if (M > HARD_MAX_LENGTH)
    {
        throw std::invalid_argument("M must be less than or equal to HARD_MAX_LENGTH, but got " + std::to_string(M) + " > " + std::to_string(HARD_MAX_LENGTH) + ".");
    }

    auto ops = torch::full({B, M}, NO_OP, expressions.options());
    auto parents = torch::full({B, M}, NULL_CHILD, expressions.options());
    auto errors = torch::zeros({B}, expressions.options().dtype(torch::kInt64));

    int64_t lparenthesis_id = get_token_id("(");
    int64_t rparenthesis_id = get_token_id(")");

    if (device.is_cuda())
    {
        parse_to_prefix_parent_cuda_impl(
            this->precedence.packed_accessor32<int64_t, 1>(),
            expressions.packed_accessor32<int64_t, 2>(),
            ops.packed_accessor32<int64_t, 2>(),
            parents.packed_accessor32<int64_t, 2>(),
            errors.packed_accessor32<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }
    else
    {
        parse_to_prefix_parent_cpu_impl(
            this->precedence.accessor<int64_t, 1>(),
            expressions.accessor<int64_t, 2>(),
            ops.accessor<int64_t, 2>(),
            parents.accessor<int64_t, 2>(),
            errors.accessor<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }

    process_parsing_errors(errors, expressions, "prefix parent parsing", verbosity);

    return std::make_tuple(ops, parents);
}

std::tuple<torch::Tensor, torch::Tensor> ProbabilisticContextFreeGrammar::parse_to_postfix(torch::Tensor expressions, int verbosity)
{
    // Check that only terminal symbols are used
    TORCH_CHECK((0 <= expressions).all().item<bool>(),
                "All symbols in the expressions must be non-negative integers.");
    TORCH_CHECK((expressions < this->terminal_limit).all().item<bool>(),
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

    process_parsing_errors(errors, expressions, "postfix parsing", verbosity);

    return std::make_tuple(ops, children);
}

std::tuple<torch::Tensor, torch::Tensor> ProbabilisticContextFreeGrammar::sample(int64_t B, int verbosity)
{
    if (B <= 0)
        throw std::invalid_argument("Batch size B must be greater than 0, but got " + std::to_string(B) + ".");

    auto string_samples = sample_string_expression(B);
    return parse_to_postfix(string_samples, verbosity);
}

std::vector<std::string> ProbabilisticContextFreeGrammar::to_string(torch::Tensor expressions)
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
            if (token_id == NO_OP) // NO_OP is padding TODO: use casting
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

std::tuple<torch::Tensor, torch::Tensor> ProbabilisticContextFreeGrammar::parse_to_postfix_parent(torch::Tensor expressions, int verbosity)
{
    // Check that only terminal symbols are used
    TORCH_CHECK((0 <= expressions).all().item<bool>(),
                "All symbols in the expressions must be non-negative integers.");
    TORCH_CHECK((expressions < this->terminal_limit).all().item<bool>(),
                "All symbols in the expressions must be terminal symbols (less than terminal_limit).");

    int64_t B = expressions.size(0);
    int64_t M = expressions.size(1);

    if (M > HARD_MAX_LENGTH)
    {
        throw std::invalid_argument("M must be less than or equal to HARD_MAX_LENGTH, but got " + std::to_string(M) + " > " + std::to_string(HARD_MAX_LENGTH) + ".");
    }

    auto ops = torch::full({B, M}, NO_OP, expressions.options());
    auto parents = torch::full({B, M}, NULL_CHILD, expressions.options());
    auto errors = torch::zeros({B}, expressions.options().dtype(torch::kInt64));

    int64_t lparenthesis_id = get_token_id("(");
    int64_t rparenthesis_id = get_token_id(")");

    if (device.is_cuda())
    {
        parse_to_postfix_parent_cuda_impl(
            this->precedence.packed_accessor32<int64_t, 1>(),
            expressions.packed_accessor32<int64_t, 2>(),
            ops.packed_accessor32<int64_t, 2>(),
            parents.packed_accessor32<int64_t, 2>(),
            errors.packed_accessor32<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }
    else
    {
        parse_to_postfix_parent_cpu_impl(
            this->precedence.accessor<int64_t, 1>(),
            expressions.accessor<int64_t, 2>(),
            ops.accessor<int64_t, 2>(),
            parents.accessor<int64_t, 2>(),
            errors.accessor<int64_t, 1>(),
            lparenthesis_id,
            rparenthesis_id,
            B,
            M);
    }

    process_parsing_errors(errors, expressions, "postfix parent parsing", verbosity);

    return std::make_tuple(ops, parents);
}

torch::Tensor ProbabilisticContextFreeGrammar::postfix_to_infix(torch::Tensor expressions, int64_t max_infix_len, int verbosity)
{
    int64_t B = expressions.size(0);
    int64_t M_postfix = expressions.size(1);

    auto infix_out = torch::full({B, max_infix_len}, NO_OP, expressions.options());
    auto errors = torch::zeros({B}, expressions.options().dtype(torch::kInt64));

    int64_t lparen_id = get_symbol_id("(");
    int64_t rparen_id = get_symbol_id(")");

    if (device.is_cuda())
    {
        postfix_to_infix_cuda_impl(
            expressions.packed_accessor32<int64_t, 2>(),
            infix_out.packed_accessor32<int64_t, 2>(),
            errors.packed_accessor32<int64_t, 1>(),
            lparen_id, rparen_id,
            B, M_postfix, max_infix_len);
    }
    else
    {
        postfix_to_infix_cpu_impl(
            expressions.accessor<int64_t, 2>(),
            infix_out.accessor<int64_t, 2>(),
            errors.accessor<int64_t, 1>(),
            this->id_to_symbol, // Not needed for tensor conversion
            lparen_id, rparen_id,
            B, M_postfix, max_infix_len);
    }

    process_parsing_errors(errors, expressions, "postfix to infix conversion", verbosity);
    return infix_out;
}

torch::Tensor ProbabilisticContextFreeGrammar::prefix_to_infix(torch::Tensor expressions, int64_t max_infix_len, int verbosity)
{
    int64_t B = expressions.size(0);
    int64_t M_prefix = expressions.size(1);

    auto infix_out = torch::full({B, max_infix_len}, NO_OP, expressions.options());
    auto errors = torch::zeros({B}, expressions.options().dtype(torch::kInt64));

    int64_t lparen_id = get_symbol_id("(");
    int64_t rparen_id = get_symbol_id(")");

    if (device.is_cuda())
    {
        prefix_to_infix_cuda_impl(
            expressions.packed_accessor32<int64_t, 2>(),
            infix_out.packed_accessor32<int64_t, 2>(),
            errors.packed_accessor32<int64_t, 1>(),
            lparen_id, rparen_id,
            B, M_prefix, max_infix_len);
    }
    else
    {
        prefix_to_infix_cpu_impl(
            expressions.accessor<int64_t, 2>(),
            infix_out.accessor<int64_t, 2>(),
            errors.accessor<int64_t, 1>(),
            lparen_id, rparen_id,
            B, M_prefix, max_infix_len);
    }

    process_parsing_errors(errors, expressions, "prefix to infix conversion", verbosity);
    return infix_out;
}

// Export using PyBind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void init_pcfg(pybind11::module &m)
{
    pybind11::class_<ProbabilisticContextFreeGrammar>(m, "ProbabilisticContextFreeGrammar")
        .def(pybind11::init<std::string, std::string, int64_t, int64_t, torch::Device, int64_t, float, bool>(),
             pybind11::arg("grammar"),
             pybind11::arg("start_symbol"),
             pybind11::arg("padded_maximum_length"),
             pybind11::arg("n_variables"),
             pybind11::arg("device"),
             pybind11::arg("max_tries") = 64,
             pybind11::arg("tolerance") = DEFAULT_tolerence,
             pybind11::arg("verbose") = false)
        .def("sample", &ProbabilisticContextFreeGrammar::sample, "Sample from the PCFG",
             pybind11::arg("B"),
             pybind11::arg("verbosity") = 0)
        .def("sample_string_expression", &ProbabilisticContextFreeGrammar::sample_string_expression, "Sample a string expression from the PCFG")
        .def("to_string", &ProbabilisticContextFreeGrammar::to_string, "Convert a tensor of expressions to a list of strings")
        .def("parse_to_postfix", &ProbabilisticContextFreeGrammar::parse_to_postfix, "Parse a string expression to a postfix representation with child pointers",
             pybind11::arg("expressions"),
             pybind11::arg("verbosity") = 0)
        .def("parse_to_postfix_parent", &ProbabilisticContextFreeGrammar::parse_to_postfix_parent, "Parse a string expression to a postfix representation with parent pointers",
             pybind11::arg("expressions"),
             pybind11::arg("verbosity") = 0)
        .def("parse_to_prefix", &ProbabilisticContextFreeGrammar::parse_to_prefix, "Parse a string expression to a prefix representation with child pointers",
             pybind11::arg("expressions"),
             pybind11::arg("verbosity") = 0)
        .def("parse_to_prefix_parent", &ProbabilisticContextFreeGrammar::parse_to_prefix_parent, "Parse a string expression to a prefix representation with parent pointers",
             pybind11::arg("expressions"),
             pybind11::arg("verbosity") = 0)
        .def("postfix_to_infix", &ProbabilisticContextFreeGrammar::postfix_to_infix, "Convert a batch of postfix expressions to infix tensors.",
             pybind11::arg("expressions"),
             pybind11::arg("max_infix_len"),
             pybind11::arg("verbosity") = 0)
        .def("prefix_to_infix", &ProbabilisticContextFreeGrammar::prefix_to_infix, "Convert a batch of prefix expressions to infix tensors.",
             pybind11::arg("expressions"),
             pybind11::arg("max_infix_len"),
             pybind11::arg("verbosity") = 0)

        .def_readonly("device", &ProbabilisticContextFreeGrammar::device)
        .def_readonly("start_symbol", &ProbabilisticContextFreeGrammar::start_symbol)
        .def_readonly("padded_maximum_length", &ProbabilisticContextFreeGrammar::padded_maximum_length)
        .def_readonly("n_variables", &ProbabilisticContextFreeGrammar::n_variables)
        .def_readonly("max_tries", &ProbabilisticContextFreeGrammar::max_tries)
        .def_readonly("tolerance", &ProbabilisticContextFreeGrammar::tolerance)
        .def_readonly("verbose", &ProbabilisticContextFreeGrammar::verbose)
        .def("get_symbol_id", &ProbabilisticContextFreeGrammar::get_symbol_id, "Get the ID of a symbol",
             pybind11::arg("symbol"))
        .def_readonly("n_operators", &ProbabilisticContextFreeGrammar::n_operators, "Get the number of operators in the grammar")
        .def("get_arities", &ProbabilisticContextFreeGrammar::get_arities, "Get the arities of the operators in the grammar")
        .def("valid_ops", &ProbabilisticContextFreeGrammar::valid_ops, "Check if the operators in the grammar are valid");
}