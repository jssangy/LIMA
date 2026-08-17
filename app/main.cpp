#include "lima/core/grid_map.hpp"
#include "lima/io/scenario_loader.hpp"
#include "lima/io/solution_trace.hpp"
#include "lima/simulation/simulator.hpp"
#include "lima/viewer/viewer.hpp"

#include "bench.hpp"
#include "lima_version.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr int kLimaDefaultProfileVersion = 5;

enum class RunMode { Realtime, Solve, Replay, Debug, Bench };

struct Options {
    std::filesystem::path map{"data/maps/cross_1.map"};
    std::filesystem::path scenario;
    std::size_t agents{15};
    std::optional<std::uint64_t> seed;
    std::uint64_t max_steps{100000};
    lima::PlannerKind planner{lima::PlannerKind::Bfs};
    bool validate_conflicts{};
    double fps{20.0};
    RunMode mode{RunMode::Realtime};
    std::filesystem::path output;
    bool no_trace{};
    std::filesystem::path replay;
    std::filesystem::path routes;
    std::filesystem::path goal_sequences;
    std::string profile{"legacy"};
    lima::SimulatorConfig sim;
    lima::bench::BenchOptions bench;
};

// Named component bundles. Profiles are applied before the ordinary CLI pass,
// so every per-component option remains an order-independent override.
void apply_profile(Options& options, const std::string_view name) {
    if (name == "legacy") {
        options.profile = "legacy";
        return;
    }
    if (name != "lima-default")
        throw std::invalid_argument("profile must be legacy or lima-default");

    options.profile = "lima-default";

    // Route Planner: Structured Waypoint Routing (SWR) with BFS segments.
    // --planner/--routing/--routes may replace this provider after the profile
    // has been applied.
    options.planner = lima::PlannerKind::Bfs;
    options.sim.direct_routing = false;
    // Local PIBT displacement may temporarily move a robot away from its
    // active route. The Route Planner reconnects it to the first unfinished
    // task-level reference waypoint while preserving the remaining suffix.
    // This uses only static-map and single-robot route information.
    options.sim.pibt_replan = 1;

    // Admission Controller: acknowledged AIMD admission window.  Gate-C
    // development cells selected beta=0.25 and additive recovery=0.50.
    options.sim.isolation = lima::IsolationConfig{};
    options.sim.isolation.formula = lima::CapacityFormula::SumMinusMax;
    options.sim.admission = lima::AdmissionConfig{};
    options.sim.admission.policy = lima::AdmissionPolicy::Aimd;
    options.sim.admission.parameter = 0.25;
    options.sim.admission.secondary = 0.50;
    options.sim.gate_resync = true;

    // Marshalling Solver: frozen beam primary plus uncut exact fallback.
    options.sim.solver = lima::SolverConfig{};
    options.sim.solver.kind = "beam-complete";
    options.sim.solver.max_iterations = 2'000'000;
    options.sim.solver.beam_width = 2'048;
    options.sim.solver.beam_score = "tt";

    // Recirculation Controller: hop-aware widest-ratio winner.
    options.sim.discharge = lima::DischargeConfig{};
    options.sim.discharge.policy = lima::DischargePolicy::WidestRatioShortest;
    options.sim.discharge_enabled = true;
}

// One line per agent: "x y x y ..." waypoint pairs; an empty line keeps the
// built-in router for that agent (CBS-timeout partial route sets).
std::vector<std::vector<lima::Coord>> load_routes(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open routes file: " + path.string());
    std::vector<std::vector<lima::Coord>> routes;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::vector<lima::Coord> route;
        int x = 0;
        int y = 0;
        while (fields >> x >> y) route.push_back({x, y});
        routes.push_back(std::move(route));
    }
    return routes;
}

void usage() {
    std::cout << "usage: lima [--profile legacy|lima-default] [--map FILE] [--scenario FILE] [--agents N] [--planner bfs|astar]"
                 " [--seed N] [--max-steps N] [--fps N] [--validate-conflicts]"
                 " [--mode realtime|solve|replay|debug] [--output FILE|--no-trace] [--replay FILE]\n"
                 "            [--solver ida|astar|wastar|gbfs|ucs|greedy|beam|beam-complete|hybrid] [--solver-iterations N]\n"
                 "            [--bound-step N] [--no-fastpath] [--lb-mode legacy|bf|tt] [--dominance]\n"
                 "            [--solver-nodes N] [--beam-width N] [--beam-score disorder|bf|tt] [--search-weight F]\n"
                 "            [--routing swr|direct] [--capacity-formula operational|plus-one] [--isolation-cap N]\n"
                 "            [--gate-policy NAME] [--gate-param F] [--gate-param2 F] [--gate-param3 F]\n"
                 "            [--admit-lookahead off|hard|thresh|ratio|diff] [--admit-lookahead-param F]\n"
                 "            [--aimd-signal local|nbmax|nbmean|trend] [--aimd-signal-param F]\n"
                 "            [--admit-credit off|equal|demand|drr] [--admit-credit-param F]\n"
                 "            [--discharge-policy NAME] [--discharge-unweighted|--discharge-random]\n"
                 "            [--discharge-partial F] [--discharge-weight F]\n"
                 "            [--recirc-probe off|detect|break-slack|break-longarm] [--recirc-probe-ttl N] [--recirc-probe-age N]\n"
                 "            [--recirc-exclusive off|id|age|reserve] [--recirc-cycle-max 4|6|8]\n"
                 "            [--no-pibt-corridor] [--pibt-sink-yield] [--pibt-arm-retreat[-last]]\n"
                 "            [--pibt-age-rate] [--pibt-replan N] [--shuffle-order SEED] [--failure-prob P]\n"
                 "            [--goal-behavior disappear|stay|lifelong] [--goal-sequences FILE]\n"
                 "            [--no-discharge] [--metrics DIR] [--trace-jsonl FILE]\n"
                 "debug mode reads commands from stdin and answers in JSON:\n"
                 "            step [n] | state | agent ID | intersection ID | summary | invariants | quit\n";
}

Options parse(const int argc, char** argv) {
    Options options;

    // Resolve the last named profile first. Explicit component flags in the
    // normal pass below then override it regardless of argument order.
    std::string_view selected_profile{"legacy"};
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) != "--profile") continue;
        if (++i >= argc) throw std::invalid_argument("missing value after --profile");
        selected_profile = argv[i];
    }
    apply_profile(options, selected_profile);

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> std::string_view {
            if (++i >= argc) throw std::invalid_argument("missing value after " + std::string(arg));
            return argv[i];
        };
        if (arg == "--profile") (void)value();
        else if (arg == "--map") options.map = value();
        else if (arg == "--scenario") options.scenario = value();
        else if (arg == "--agents") options.agents = std::stoull(std::string(value()));
        else if (arg == "--seed") options.seed = std::stoull(std::string(value()));
        else if (arg == "--max-steps") options.max_steps = std::stoull(std::string(value()));
        else if (arg == "--fps") options.fps = std::stod(std::string(value()));
        else if (arg == "--validate-conflicts") options.validate_conflicts = true;
        else if (arg == "--output") options.output = value();
        else if (arg == "--no-trace") options.no_trace = true;
        else if (arg == "--replay") options.replay = value();
        else if (arg == "--routes") options.routes = value();
        else if (arg == "--goal-sequences") options.goal_sequences = value();
        else if (arg == "--solver") options.sim.solver.kind = std::string(value());
        else if (arg == "--solver-iterations") options.sim.solver.max_iterations = std::stoull(std::string(value()));
        else if (arg == "--bound-step") options.sim.solver.bound_step = std::stoi(std::string(value()));
        else if (arg == "--no-fastpath") options.sim.solver.greedy_fastpath = false;
        else if (arg == "--solver-max-capacity") options.sim.solver.max_capacity = std::stoi(std::string(value()));
        else if (arg == "--lb-mode") options.sim.solver.lb_mode = std::string(value());
        else if (arg == "--dominance") options.sim.solver.dominance = true;
        else if (arg == "--solver-nodes") options.sim.solver.max_nodes = std::stoull(std::string(value()));
        else if (arg == "--beam-width") options.sim.solver.beam_width = std::stoull(std::string(value()));
        else if (arg == "--beam-score") options.sim.solver.beam_score = std::string(value());
        else if (arg == "--search-weight") options.sim.solver.best_first_weight = std::stod(std::string(value()));
        else if (arg == "--isolation-cap") options.sim.isolation.cap = std::stoi(std::string(value()));
        else if (arg == "--isolation-margin") options.sim.isolation.margin = std::stoi(std::string(value()));
        else if (arg == "--stall-threshold") options.sim.stall_threshold = static_cast<std::uint32_t>(std::stoul(std::string(value())));
        else if (arg == "--discharge-all-arms") options.sim.discharge.all_arms = true;
        else if (arg == "--discharge-stalled-neighbor") options.sim.discharge.allow_stalled_neighbor = true;
        else if (arg == "--discharge-deterministic") options.sim.discharge.deterministic_cycle = true;
        else if (arg == "--discharge-avail-weighted") options.sim.discharge.avail_weighted = true;
        else if (arg == "--discharge-unweighted") {
            options.sim.discharge.deterministic_cycle = true;
            options.sim.discharge.avail_weighted = false;
        }
        else if (arg == "--discharge-random") {
            options.sim.discharge.deterministic_cycle = false;
            options.sim.discharge.avail_weighted = false;
        }
        else if (arg == "--discharge-partial") options.sim.discharge.partial_stall = std::stod(std::string(value()));
        else if (arg == "--discharge-policy") {
            const auto name = value();
            if (name == "composite") options.sim.discharge.policy = lima::DischargePolicy::Composite;
            else if (name == "random") options.sim.discharge.policy = lima::DischargePolicy::Random;
            else if (name == "least-load") options.sim.discharge.policy = lima::DischargePolicy::LeastLoaded;
            else if (name == "max-slack") options.sim.discharge.policy = lima::DischargePolicy::MaxSlack;
            else if (name == "rotor") options.sim.discharge.policy = lima::DischargePolicy::Rotor;
            else if (name == "shortest") options.sim.discharge.policy = lima::DischargePolicy::Shortest;
            else if (name == "power-two") options.sim.discharge.policy = lima::DischargePolicy::PowerOfTwo;
            else if (name == "backpressure") options.sim.discharge.policy = lima::DischargePolicy::Backpressure;
            else if (name == "balanced") options.sim.discharge.policy = lima::DischargePolicy::Balanced;
            else if (name == "demand") options.sim.discharge.policy = lima::DischargePolicy::Demand;
            else if (name == "widest-ratio-shortest") options.sim.discharge.policy = lima::DischargePolicy::WidestRatioShortest;
            else throw std::invalid_argument("unknown discharge policy: " + std::string(name));
        }
        else if (arg == "--discharge-weight") {
            options.sim.discharge.weight = std::stod(std::string(value()));
            if (options.sim.discharge.weight < 0.0)
                throw std::invalid_argument("discharge-weight must be non-negative");
        }
        else if (arg == "--recirc-probe") {
            const auto name = value();
            if (name == "off") options.sim.discharge.probe.mode = lima::RecirculationProbeMode::Off;
            else if (name == "detect") options.sim.discharge.probe.mode = lima::RecirculationProbeMode::Detect;
            else if (name == "break-slack") options.sim.discharge.probe.mode = lima::RecirculationProbeMode::BreakAtSlack;
            else if (name == "break-longarm") options.sim.discharge.probe.mode = lima::RecirculationProbeMode::BreakAtLongArm;
            else throw std::invalid_argument("unknown recirc-probe: " + std::string(name));
        }
        else if (arg == "--recirc-probe-ttl")
            options.sim.discharge.probe.ttl = static_cast<std::uint16_t>(std::stoul(std::string(value())));
        else if (arg == "--recirc-probe-age")
            options.sim.discharge.probe.activation_age = static_cast<std::uint16_t>(std::stoul(std::string(value())));
        else if (arg == "--recirc-exclusive") {
            const auto name = value();
            if (name == "off") options.sim.discharge.exclusive = lima::RecirculationExclusiveMode::Off;
            else if (name == "id") options.sim.discharge.exclusive = lima::RecirculationExclusiveMode::IntersectionId;
            else if (name == "age") options.sim.discharge.exclusive = lima::RecirculationExclusiveMode::StallAge;
            else if (name == "reserve") options.sim.discharge.exclusive = lima::RecirculationExclusiveMode::ReserveCells;
            else throw std::invalid_argument("unknown recirc-exclusive: " + std::string(name));
        }
        else if (arg == "--recirc-cycle-max") {
            const auto maximum = std::stoul(std::string(value()));
            if (maximum != 4 && maximum != 6 && maximum != 8)
                throw std::invalid_argument("recirc-cycle-max must be 4, 6, or 8");
            options.sim.discharge.cycle_max = static_cast<std::uint8_t>(maximum);
        }
        else if (arg == "--admit-hysteresis") options.sim.isolation.hysteresis = std::stoi(std::string(value()));
        else if (arg == "--gate-policy") {
            const auto name = value();
            if (name == "static") options.sim.admission.policy = lima::AdmissionPolicy::Static;
            else if (name == "fraction") options.sim.admission.policy = lima::AdmissionPolicy::FractionalReserve;
            else if (name == "request") options.sim.admission.policy = lima::AdmissionPolicy::RequestProportional;
            else if (name == "backpressure") options.sim.admission.policy = lima::AdmissionPolicy::Backpressure;
            else if (name == "neighbor-pressure") options.sim.admission.policy = lima::AdmissionPolicy::NeighborPressure;
            else if (name == "aimd") options.sim.admission.policy = lima::AdmissionPolicy::Aimd;
            else if (name == "red") options.sim.admission.policy = lima::AdmissionPolicy::Red;
            else if (name == "blue") options.sim.admission.policy = lima::AdmissionPolicy::Blue;
            else if (name == "rem") options.sim.admission.policy = lima::AdmissionPolicy::Rem;
            else if (name == "avq") options.sim.admission.policy = lima::AdmissionPolicy::Avq;
            else if (name == "codel") options.sim.admission.policy = lima::AdmissionPolicy::Codel;
            else if (name == "pi") options.sim.admission.policy = lima::AdmissionPolicy::Pi;
            else if (name == "pie") options.sim.admission.policy = lima::AdmissionPolicy::Pie;
            else if (name == "token") options.sim.admission.policy = lima::AdmissionPolicy::TokenBucket;
            else if (name == "sotl") options.sim.admission.policy = lima::AdmissionPolicy::Sotl;
            else if (name == "choke") options.sim.admission.policy = lima::AdmissionPolicy::Choke;
            else if (name == "queue-csma") options.sim.admission.policy = lima::AdmissionPolicy::QueueCsma;
            else if (name == "sfb") options.sim.admission.policy = lima::AdmissionPolicy::StochasticFairBlue;
            else if (name == "fq-codel") options.sim.admission.policy = lima::AdmissionPolicy::FqCodel;
            else if (name == "lqf") options.sim.admission.policy = lima::AdmissionPolicy::LongestQueue;
            else if (name == "oldest") options.sim.admission.policy = lima::AdmissionPolicy::OldestRequest;
            else if (name == "round-robin") options.sim.admission.policy = lima::AdmissionPolicy::RoundRobin;
            else if (name == "drr") options.sim.admission.policy = lima::AdmissionPolicy::DeficitRoundRobin;
            else throw std::invalid_argument("unknown gate-policy: " + std::string(name));
        }
        else if (arg == "--gate-param") options.sim.admission.parameter = std::stod(std::string(value()));
        else if (arg == "--gate-param2") options.sim.admission.secondary = std::stod(std::string(value()));
        else if (arg == "--gate-param3") options.sim.admission.tertiary = std::stod(std::string(value()));
        else if (arg == "--admit-lookahead") {
            const auto name = value();
            if (name == "off") options.sim.admission_information.lookahead = lima::AdmitLookaheadMode::Off;
            else if (name == "hard") options.sim.admission_information.lookahead = lima::AdmitLookaheadMode::Hard;
            else if (name == "thresh") options.sim.admission_information.lookahead = lima::AdmitLookaheadMode::Threshold;
            else if (name == "ratio") options.sim.admission_information.lookahead = lima::AdmitLookaheadMode::Ratio;
            else if (name == "diff") options.sim.admission_information.lookahead = lima::AdmitLookaheadMode::Differential;
            else throw std::invalid_argument("unknown admit-lookahead: " + std::string(name));
        }
        else if (arg == "--admit-lookahead-param")
            options.sim.admission_information.lookahead_parameter = std::stod(std::string(value()));
        else if (arg == "--aimd-signal") {
            const auto name = value();
            if (name == "local") options.sim.admission_information.aimd_signal = lima::AimdSignalMode::Local;
            else if (name == "nbmax") options.sim.admission_information.aimd_signal = lima::AimdSignalMode::NeighborMax;
            else if (name == "nbmean") options.sim.admission_information.aimd_signal = lima::AimdSignalMode::NeighborMean;
            else if (name == "trend") options.sim.admission_information.aimd_signal = lima::AimdSignalMode::Trend;
            else throw std::invalid_argument("unknown aimd-signal: " + std::string(name));
        }
        else if (arg == "--aimd-signal-param")
            options.sim.admission_information.aimd_signal_parameter = std::stod(std::string(value()));
        else if (arg == "--admit-credit") {
            const auto name = value();
            if (name == "off") options.sim.admission_information.credit = lima::AdmitCreditMode::Off;
            else if (name == "equal") options.sim.admission_information.credit = lima::AdmitCreditMode::Equal;
            else if (name == "demand") options.sim.admission_information.credit = lima::AdmitCreditMode::Demand;
            else if (name == "drr") options.sim.admission_information.credit = lima::AdmitCreditMode::DeficitRoundRobin;
            else throw std::invalid_argument("unknown admit-credit: " + std::string(name));
        }
        else if (arg == "--admit-credit-param")
            options.sim.admission_information.credit_parameter = std::stod(std::string(value()));
        else if (arg == "--no-discharge") options.sim.discharge_enabled = false;
        else if (arg == "--rotation") options.sim.rotation_enabled = true;
        else if (arg == "--gate-resync") options.sim.gate_resync = true;
        else if (arg == "--no-gate-resync") options.sim.gate_resync = false;
        else if (arg == "--subset-scheduling") options.sim.subset_scheduling = true;
        else if (arg == "--pibt-corridor") options.sim.pibt_corridor = true;
        else if (arg == "--no-pibt-corridor") options.sim.pibt_corridor = false;
        else if (arg == "--pibt-sink-yield") options.sim.pibt_sink_yield = true;
        else if (arg == "--pibt-arm-retreat") options.sim.pibt_arm_retreat = true;
        else if (arg == "--pibt-arm-retreat-last") options.sim.pibt_arm_retreat_last = true;
        else if (arg == "--pibt-age-rate") options.sim.pibt_age_rate = true;
        else if (arg == "--pibt-replan") options.sim.pibt_replan = static_cast<std::uint32_t>(std::stoul(std::string(value())));
        else if (arg == "--shuffle-order") options.sim.shuffle_order = std::stoll(std::string(value()));
        else if (arg == "--failure-prob") {
            options.sim.failure_probability = std::stod(std::string(value()));
            if (options.sim.failure_probability < 0.0 || options.sim.failure_probability > 1.0)
                throw std::invalid_argument("failure-prob must be in [0,1]");
        }
        else if (arg == "--goal-behavior") {
            const auto name = value();
            if (name == "disappear") options.sim.goal_behavior = lima::GoalBehavior::Disappear;
            else if (name == "stay") options.sim.goal_behavior = lima::GoalBehavior::Stay;
            else if (name == "lifelong") options.sim.goal_behavior = lima::GoalBehavior::Lifelong;
            else throw std::invalid_argument("goal-behavior must be disappear, stay, or lifelong");
        }
        else if (arg == "--metrics") options.sim.metrics_dir = std::string(value());
        else if (arg == "--trace-jsonl") options.sim.trace_path = std::string(value());
        else if (arg == "--bench-arms") {
            options.bench.capacities.clear();
            std::string text(value());
            std::replace(text.begin(), text.end(), ',', ' ');
            std::istringstream arms(text);
            for (int capacity = 0; arms >> capacity;) options.bench.capacities.push_back(capacity);
            if (options.bench.capacities.empty() || options.bench.capacities.size() > 4)
                throw std::invalid_argument("--bench-arms takes 1..4 comma-separated capacities");
        }
        else if (arg == "--bench-n") options.bench.items = std::stoi(std::string(value()));
        else if (arg == "--bench-instances") options.bench.instances = std::stoi(std::string(value()));
        else if (arg == "--routing") {
            const auto name = value();
            if (name == "swr" || name == "dor") options.sim.direct_routing = false;
            else if (name == "direct") options.sim.direct_routing = true;
            else throw std::invalid_argument("routing must be swr or direct");
        }
        else if (arg == "--capacity-formula") {
            const auto name = value();
            if (name == "operational" || name == "code")
                options.sim.isolation.formula = lima::CapacityFormula::SumMinusMax;
            else if (name == "plus-one" || name == "paper")
                options.sim.isolation.formula = lima::CapacityFormula::SumMinusMaxPlusOne;
            else throw std::invalid_argument("capacity-formula must be operational or plus-one");
        }
        else if (arg == "--mode") {
            const auto name = value();
            if (name == "realtime") options.mode = RunMode::Realtime;
            else if (name == "solve") options.mode = RunMode::Solve;
            else if (name == "replay") options.mode = RunMode::Replay;
            else if (name == "debug") options.mode = RunMode::Debug;
            else if (name == "bench") options.mode = RunMode::Bench;
            else throw std::invalid_argument("mode must be realtime, solve, replay, debug, or bench");
        }
        else if (arg == "--planner") {
            const auto name = value();
            if (name == "bfs") options.planner = lima::PlannerKind::Bfs;
            else if (name == "astar") options.planner = lima::PlannerKind::AStar;
            else throw std::invalid_argument("planner must be bfs or astar");
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + std::string(arg));
    }
    return options;
}

std::uint64_t random_seed() {
    std::random_device device;
    const auto now = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return (static_cast<std::uint64_t>(device()) << 32U)
        ^ static_cast<std::uint64_t>(device()) ^ now;
}

std::string_view admission_policy_name(const lima::AdmissionPolicy policy) {
    switch (policy) {
    case lima::AdmissionPolicy::Static: return "static";
    case lima::AdmissionPolicy::FractionalReserve: return "fraction";
    case lima::AdmissionPolicy::RequestProportional: return "request";
    case lima::AdmissionPolicy::Backpressure: return "backpressure";
    case lima::AdmissionPolicy::NeighborPressure: return "neighbor-pressure";
    case lima::AdmissionPolicy::Aimd: return "aimd";
    case lima::AdmissionPolicy::Red: return "red";
    case lima::AdmissionPolicy::Blue: return "blue";
    case lima::AdmissionPolicy::Rem: return "rem";
    case lima::AdmissionPolicy::Avq: return "avq";
    case lima::AdmissionPolicy::Codel: return "codel";
    case lima::AdmissionPolicy::Pi: return "pi";
    case lima::AdmissionPolicy::Pie: return "pie";
    case lima::AdmissionPolicy::TokenBucket: return "token";
    case lima::AdmissionPolicy::Sotl: return "sotl";
    case lima::AdmissionPolicy::Choke: return "choke";
    case lima::AdmissionPolicy::QueueCsma: return "queue-csma";
    case lima::AdmissionPolicy::StochasticFairBlue: return "sfb";
    case lima::AdmissionPolicy::FqCodel: return "fq-codel";
    case lima::AdmissionPolicy::LongestQueue: return "lqf";
    case lima::AdmissionPolicy::OldestRequest: return "oldest";
    case lima::AdmissionPolicy::RoundRobin: return "round-robin";
    case lima::AdmissionPolicy::DeficitRoundRobin: return "drr";
    }
    return "unknown";
}

std::string_view discharge_policy_name(const lima::DischargePolicy policy) {
    switch (policy) {
    case lima::DischargePolicy::Legacy: return "legacy";
    case lima::DischargePolicy::Composite: return "composite";
    case lima::DischargePolicy::Random: return "random";
    case lima::DischargePolicy::LeastLoaded: return "least-load";
    case lima::DischargePolicy::MaxSlack: return "max-slack";
    case lima::DischargePolicy::Rotor: return "rotor";
    case lima::DischargePolicy::Shortest: return "shortest";
    case lima::DischargePolicy::PowerOfTwo: return "power-two";
    case lima::DischargePolicy::Backpressure: return "backpressure";
    case lima::DischargePolicy::Balanced: return "balanced";
    case lima::DischargePolicy::Demand: return "demand";
    case lima::DischargePolicy::WidestRatioShortest: return "widest-ratio-shortest";
    }
    return "unknown";
}

std::string_view admit_lookahead_name(const lima::AdmitLookaheadMode mode) {
    switch (mode) {
    case lima::AdmitLookaheadMode::Off: return "off";
    case lima::AdmitLookaheadMode::Hard: return "hard";
    case lima::AdmitLookaheadMode::Threshold: return "thresh";
    case lima::AdmitLookaheadMode::Ratio: return "ratio";
    case lima::AdmitLookaheadMode::Differential: return "diff";
    }
    return "unknown";
}

std::string_view aimd_signal_name(const lima::AimdSignalMode mode) {
    switch (mode) {
    case lima::AimdSignalMode::Local: return "local";
    case lima::AimdSignalMode::NeighborMax: return "nbmax";
    case lima::AimdSignalMode::NeighborMean: return "nbmean";
    case lima::AimdSignalMode::Trend: return "trend";
    }
    return "unknown";
}

std::string_view admit_credit_name(const lima::AdmitCreditMode mode) {
    switch (mode) {
    case lima::AdmitCreditMode::Off: return "off";
    case lima::AdmitCreditMode::Equal: return "equal";
    case lima::AdmitCreditMode::Demand: return "demand";
    case lima::AdmitCreditMode::DeficitRoundRobin: return "drr";
    }
    return "unknown";
}

std::string_view recirc_probe_name(const lima::RecirculationProbeMode mode) {
    switch (mode) {
    case lima::RecirculationProbeMode::Off: return "off";
    case lima::RecirculationProbeMode::Detect: return "detect";
    case lima::RecirculationProbeMode::BreakAtSlack: return "break-slack";
    case lima::RecirculationProbeMode::BreakAtLongArm: return "break-longarm";
    }
    return "unknown";
}

std::string_view recirc_exclusive_name(const lima::RecirculationExclusiveMode mode) {
    switch (mode) {
    case lima::RecirculationExclusiveMode::Off: return "off";
    case lima::RecirculationExclusiveMode::IntersectionId: return "id";
    case lima::RecirculationExclusiveMode::StallAge: return "age";
    case lima::RecirculationExclusiveMode::ReserveCells: return "reserve";
    }
    return "unknown";
}

std::string_view planner_name(const lima::PlannerKind planner) {
    switch (planner) {
    case lima::PlannerKind::Bfs: return "bfs";
    case lima::PlannerKind::AStar: return "astar";
    }
    return "unknown";
}

bool has_non_default_config(const Options& options) {
    const lima::SolverConfig defaults;
    return options.profile != "legacy"
        || options.planner != lima::PlannerKind::Bfs
        || !options.routes.empty()
        || !options.goal_sequences.empty()
        || options.sim.solver.kind != defaults.kind
        || options.sim.solver.max_iterations != defaults.max_iterations
        || options.sim.solver.bound_step != defaults.bound_step
        || options.sim.solver.greedy_fastpath != defaults.greedy_fastpath
        || options.sim.solver.max_capacity != defaults.max_capacity
        || options.sim.solver.lb_mode != defaults.lb_mode
        || options.sim.solver.dominance != defaults.dominance
        || options.sim.solver.max_nodes != defaults.max_nodes
        || options.sim.solver.beam_width != defaults.beam_width
        || options.sim.solver.beam_score != defaults.beam_score
        || options.sim.solver.best_first_weight != defaults.best_first_weight
        || options.sim.isolation.formula != lima::CapacityFormula::SumMinusMax
        || options.sim.isolation.cap >= 0
        || options.sim.isolation.margin != 0
        || options.sim.stall_threshold != 10
        || options.sim.discharge.all_arms
        || options.sim.discharge.allow_stalled_neighbor
        || !options.sim.discharge.deterministic_cycle
        || !options.sim.discharge.avail_weighted
        || options.sim.discharge.partial_stall != 1.0
        || options.sim.discharge.policy != lima::DischargePolicy::Legacy
        || options.sim.discharge.weight != 1.0
        || options.sim.discharge.probe.mode != lima::RecirculationProbeMode::Off
        || options.sim.discharge.exclusive != lima::RecirculationExclusiveMode::Off
        || options.sim.discharge.cycle_max != 4
        || options.sim.isolation.hysteresis != 0
        || options.sim.admission.policy != lima::AdmissionPolicy::Static
        || options.sim.admission_information.lookahead != lima::AdmitLookaheadMode::Off
        || options.sim.admission_information.aimd_signal != lima::AimdSignalMode::Local
        || options.sim.admission_information.credit != lima::AdmitCreditMode::Off
        || options.sim.rotation_enabled
        || !options.sim.gate_resync
        || options.sim.subset_scheduling
        || !options.sim.pibt_corridor
        || options.sim.pibt_sink_yield
        || options.sim.pibt_arm_retreat
        || options.sim.pibt_arm_retreat_last
        || options.sim.pibt_age_rate
        || options.sim.pibt_replan != 0
        || options.sim.shuffle_order >= 0
        || options.sim.failure_probability != 0.0
        || options.sim.goal_behavior != lima::GoalBehavior::Disappear
        || !options.sim.discharge_enabled
        || options.sim.direct_routing
        || !options.sim.metrics_dir.empty()
        || !options.sim.trace_path.empty()
        || options.no_trace;
}

// Keeps default-flag stdout byte-compatible with the submitted simulator:
// the provenance suffix appears only when a non-default knob is active.
void print_provenance(const Options& options) {
    if (!has_non_default_config(options)) return;
    if (options.profile != "legacy")
        std::cout << " profile=" << options.profile
                  << " profile_version=" << kLimaDefaultProfileVersion;
    if (options.profile != "legacy" || options.planner != lima::PlannerKind::Bfs || !options.routes.empty()) {
        std::cout << " planner=";
        if (!options.routes.empty()) std::cout << "external+";
        std::cout << planner_name(options.planner);
    }
    std::cout << " solver=" << options.sim.solver.kind
              << " routing=" << (options.sim.direct_routing ? "direct" : "swr")
              << " capacity=" << (options.sim.isolation.formula == lima::CapacityFormula::SumMinusMax
                  ? "operational" : "plus-one");
    if (options.profile != "legacy"
        || options.sim.solver.max_iterations != lima::SolverConfig{}.max_iterations)
        std::cout << " solver_iterations=" << options.sim.solver.max_iterations;
    if (options.sim.solver.lb_mode != "legacy") std::cout << " lb=" << options.sim.solver.lb_mode;
    if (options.sim.solver.dominance) std::cout << " dom=on";
    if (options.profile != "legacy"
        || options.sim.solver.beam_width != lima::SolverConfig{}.beam_width)
        std::cout << " beam_width=" << options.sim.solver.beam_width;
    if (options.profile != "legacy"
        || options.sim.solver.beam_score != lima::SolverConfig{}.beam_score)
        std::cout << " beam_score=" << options.sim.solver.beam_score;
    if (options.sim.solver.kind == "beam-complete")
        std::cout << " fallback=ida-exact";
    if (options.sim.solver.best_first_weight != lima::SolverConfig{}.best_first_weight)
        std::cout << " search_weight=" << options.sim.solver.best_first_weight;
    if (options.sim.isolation.cap >= 0) std::cout << " cap=" << options.sim.isolation.cap;
    if (options.sim.isolation.margin != 0) std::cout << " margin=" << options.sim.isolation.margin;
    if (options.sim.isolation.hysteresis != 0) std::cout << " hysteresis=" << options.sim.isolation.hysteresis;
    if (options.profile != "legacy" || options.sim.admission.policy != lima::AdmissionPolicy::Static) {
        std::cout << " gate_policy=" << admission_policy_name(options.sim.admission.policy);
        if (options.sim.admission.policy != lima::AdmissionPolicy::Static) {
            if (options.sim.admission.parameter != 0.0) std::cout << " gate_param=" << options.sim.admission.parameter;
            if (options.sim.admission.secondary != 0.0) std::cout << " gate_param2=" << options.sim.admission.secondary;
            if (options.sim.admission.tertiary != 0.0) std::cout << " gate_param3=" << options.sim.admission.tertiary;
        }
    }
    if (options.profile != "legacy" || !options.sim.gate_resync)
        std::cout << " gate_resync=" << (options.sim.gate_resync ? "on" : "off");
    if (options.sim.admission_information.lookahead != lima::AdmitLookaheadMode::Off) {
        std::cout << " admit_lookahead="
                  << admit_lookahead_name(options.sim.admission_information.lookahead)
                  << " admit_lookahead_param="
                  << options.sim.admission_information.lookahead_parameter;
    }
    if (options.sim.admission_information.aimd_signal != lima::AimdSignalMode::Local) {
        std::cout << " aimd_signal="
                  << aimd_signal_name(options.sim.admission_information.aimd_signal)
                  << " aimd_signal_param="
                  << options.sim.admission_information.aimd_signal_parameter;
    }
    if (options.sim.admission_information.credit != lima::AdmitCreditMode::Off) {
        std::cout << " admit_credit="
                  << admit_credit_name(options.sim.admission_information.credit);
        if (options.sim.admission_information.credit == lima::AdmitCreditMode::DeficitRoundRobin)
            std::cout << " admit_credit_param="
                      << options.sim.admission_information.credit_parameter;
    }
    if (options.sim.subset_scheduling) std::cout << " subset=on";
    if (!options.sim.pibt_corridor) std::cout << " pibt=off";
    if (options.sim.pibt_sink_yield) std::cout << " sink_yield=on";
    if (options.sim.pibt_arm_retreat) std::cout << " arm_retreat=on";
    if (options.sim.pibt_arm_retreat_last) std::cout << " arm_retreat=last";
    if (options.sim.pibt_age_rate) std::cout << " age_rate=on";
    if (options.sim.pibt_replan != 0) std::cout << " replan=" << options.sim.pibt_replan;
    if (options.sim.shuffle_order >= 0) std::cout << " shuffle=" << options.sim.shuffle_order;
    if (options.sim.failure_probability != 0.0)
        std::cout << " failure_prob=" << options.sim.failure_probability
                  << " delay_trace=counter-hash-v1";
    if (options.profile != "legacy" || options.sim.discharge.policy != lima::DischargePolicy::Legacy) {
        std::cout << " discharge_policy=" << discharge_policy_name(options.sim.discharge.policy);
        if (options.sim.discharge.policy == lima::DischargePolicy::Balanced)
            std::cout << " discharge_weight=" << options.sim.discharge.weight;
    } else if (!options.sim.discharge.deterministic_cycle)
        std::cout << " discharge_policy=random";
    else if (!options.sim.discharge.avail_weighted)
        std::cout << " discharge_policy=load_only";
    if (options.sim.discharge.all_arms) std::cout << " discharge_all_arms=on";
    if (options.sim.discharge.allow_stalled_neighbor) std::cout << " discharge_stalled_neighbor=on";
    if (options.sim.discharge.partial_stall != 1.0)
        std::cout << " discharge_partial=" << options.sim.discharge.partial_stall;
    if (options.sim.discharge.probe.mode != lima::RecirculationProbeMode::Off)
        std::cout << " recirc_probe=" << recirc_probe_name(options.sim.discharge.probe.mode)
                  << " recirc_probe_ttl=" << options.sim.discharge.probe.ttl
                  << " recirc_probe_age=" << options.sim.discharge.probe.activation_age;
    if (options.sim.discharge.exclusive != lima::RecirculationExclusiveMode::Off)
        std::cout << " recirc_exclusive="
                  << recirc_exclusive_name(options.sim.discharge.exclusive);
    if (options.sim.discharge.cycle_max != 4)
        std::cout << " recirc_cycle_max=" << static_cast<int>(options.sim.discharge.cycle_max);
    if (options.profile != "legacy" && options.sim.goal_behavior == lima::GoalBehavior::Disappear)
        std::cout << " goal_behavior=disappear";
    else if (options.sim.goal_behavior == lima::GoalBehavior::Stay)
        std::cout << " goal_behavior=stay";
    else if (options.sim.goal_behavior == lima::GoalBehavior::Lifelong)
        std::cout << " goal_behavior=lifelong";
    if (!options.goal_sequences.empty()) std::cout << " goal_sequences=fixed-cyclic";
    if (options.profile != "legacy" || !options.sim.discharge_enabled)
        std::cout << " discharge=" << (options.sim.discharge_enabled ? "on" : "off");
    if (options.no_trace) std::cout << " trace=off";
    std::cout << " commit=" << LIMA_COMMIT;
}

constexpr std::array<std::string_view, 9> kWaitReasonNames{
    "none", "scheduled_hold", "intersection_reserved", "intersection_capacity",
    "vertex_conflict", "edge_swap", "dependency", "schedule_group", "execution_failure"};

// Interactive JSON REPL so an AI agent can advance the simulation one step at
// a time and verify state and invariants programmatically.
int run_debug(lima::Simulator& simulator, const Options& options, lima::SolutionTrace* recorder) {
    const auto& map = simulator.map();
    const auto coord_json = [&](const lima::CellId cell) {
        std::ostringstream text;
        const lima::Coord c = map.coord(cell);
        text << '[' << c.x << ',' << c.y << ']';
        return text.str();
    };
    const auto agent_json = [&](const lima::Agent& agent) {
        std::ostringstream text;
        text << "{\"id\":" << agent.id << ",\"pos\":" << coord_json(agent.position)
             << ",\"cell\":" << agent.position
             << ",\"goal\":" << coord_json(agent.goal)
             << ",\"active\":" << (agent.active ? "true" : "false")
             << ",\"scheduled\":" << (agent.scheduled() ? "true" : "false")
             << ",\"awaiting_goal\":" << (agent.awaiting_goal ? "true" : "false")
             << ",\"tasks_completed\":" << agent.tasks_completed
             << ",\"wait_steps\":" << agent.wait_steps
             << ",\"wait_reason\":\"" << kWaitReasonNames[static_cast<std::size_t>(agent.wait_reason)] << '"'
             << ",\"moves\":" << agent.moves
             << ",\"route_remaining\":" << (agent.route.size() - 1 - agent.route_cursor)
             << ",\"next\":" << coord_json(agent.intended_cell()) << '}';
        return text.str();
    };
    const auto summary_json = [&]() {
        const auto& stats = simulator.stats();
        std::ostringstream text;
        text << "{\"t\":" << stats.timestep << ",\"completed\":" << stats.completed
             << ",\"agents\":" << simulator.agents().size()
             << ",\"moves\":" << stats.committed_moves << ",\"waits\":" << stats.waits
             << ",\"deadlocks\":" << stats.detected_deadlocks
             << ",\"done\":" << (simulator.done() ? "true" : "false") << '}';
        return text.str();
    };

    std::cout << summary_json() << '\n' << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command.empty()) continue;
        if (command == "quit" || command == "exit") break;
        if (command == "step") {
            std::int64_t parsed = 1;
            if (!(input >> parsed) && !input.eof()) {
                std::cout << "{\"error\":\"bad step count\"}\n" << std::flush;
                continue;
            }
            if (parsed < 0) {
                std::cout << "{\"error\":\"step count must be positive\"}\n" << std::flush;
                continue;
            }
            std::uint64_t requested = parsed == 0 ? 1 : static_cast<std::uint64_t>(parsed);
            std::uint64_t advanced = 0;
            bool alive = true;
            while (advanced < requested && !simulator.done()
                   && simulator.stats().timestep < options.max_steps) {
                alive = simulator.step();
                if (!alive) break;
                if (recorder) recorder->append(simulator.agents());
                ++advanced;
            }
            std::cout << "{\"advanced\":" << advanced << ",\"stalled\":" << (alive ? "false" : "true")
                      << ",\"summary\":" << summary_json() << "}\n" << std::flush;
        } else if (command == "state") {
            std::cout << "{\"summary\":" << summary_json() << ",\"agents\":[";
            const auto& agents = simulator.agents();
            for (std::size_t i = 0; i < agents.size(); ++i) {
                if (i > 0) std::cout << ',';
                std::cout << agent_json(agents[i]);
            }
            std::cout << "]}\n" << std::flush;
        } else if (command == "agent") {
            std::int64_t id = -1;
            if (!(input >> id)) {
                std::cout << "{\"error\":\"unknown agent\"}\n" << std::flush;
                continue;
            }
            const auto& agents = simulator.agents();
            if (id < 0 || static_cast<std::size_t>(id) >= agents.size()) {
                std::cout << "{\"error\":\"unknown agent\"}\n" << std::flush;
            } else {
                std::cout << agent_json(agents[static_cast<std::size_t>(id)]) << '\n' << std::flush;
            }
        } else if (command == "intersection") {
            std::int64_t id = -1;
            if (!(input >> id)) {
                std::cout << "{\"error\":\"unknown intersection\"}\n" << std::flush;
                continue;
            }
            const auto& intersections = simulator.topology().intersections();
            if (id < 0 || static_cast<std::size_t>(id) >= intersections.size()) {
                std::cout << "{\"error\":\"unknown intersection\"}\n" << std::flush;
                continue;
            }
            const auto& intersection = intersections[static_cast<std::size_t>(id)];
            const auto iid = static_cast<std::size_t>(id);
            std::cout << "{\"id\":" << id << ",\"center\":" << coord_json(intersection.center)
                      << ",\"arms\":[" << intersection.arms[0].size() << ',' << intersection.arms[1].size()
                      << ',' << intersection.arms[2].size() << ',' << intersection.arms[3].size() << ']'
                      << ",\"capacity\":" << simulator.intersection_capacity()[iid]
                      << ",\"available\":" << simulator.intersection_available()[iid]
                      << ",\"active\":" << (simulator.deadlock_active()[iid] != 0 ? "true" : "false")
                      << ",\"waiting\":" << (simulator.deadlock_waiting()[iid] ? "true" : "false")
                      << ",\"members\":[";
            bool first = true;
            for (const lima::Agent& agent : simulator.agents()) {
                if (!agent.active) continue;
                const auto& memberships = simulator.topology().memberships(agent.position);
                if (std::find(memberships.begin(), memberships.end(),
                              static_cast<lima::IntersectionId>(id)) == memberships.end()) continue;
                if (!first) std::cout << ',';
                std::cout << agent.id;
                first = false;
            }
            std::cout << "]}\n" << std::flush;
        } else if (command == "summary") {
            std::cout << summary_json() << '\n' << std::flush;
        } else if (command == "invariants") {
            const std::string violation = simulator.check_invariants();
            if (violation.empty()) std::cout << "{\"ok\":true}\n" << std::flush;
            else std::cout << "{\"ok\":false,\"violation\":\"" << violation << "\"}\n" << std::flush;
        } else {
            std::cout << "{\"error\":\"unknown command\",\"commands\":[\"step\",\"state\",\"agent\",\"intersection\",\"summary\",\"invariants\",\"quit\"]}\n" << std::flush;
        }
    }
    simulator.write_metrics();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--version") {
            std::cout << "commit=" << LIMA_COMMIT
                      << " profile=lima-default"
                      << " profile_version=" << kLimaDefaultProfileVersion << '\n';
            return 0;
        }
        Options options = parse(argc, argv);
        if (options.mode == RunMode::Bench) {
            options.bench.seed = options.seed.value_or(random_seed());
            options.bench.csv_path = options.output.string();
            // Stress instances may exceed the baseline acceptance bound on purpose.
            for (const int capacity : options.bench.capacities)
                options.sim.solver.max_capacity = std::max(options.sim.solver.max_capacity, capacity);
            const auto solver = lima::make_solver(options.sim.solver);
            return lima::bench::run(*solver, options.bench);
        }
        options.sim.map_file = options.map.string();
        lima::GridMap map = lima::GridMap::load(options.map);
        if (options.fps < 0.0) throw std::invalid_argument("fps must be zero or greater");

        if (options.mode == RunMode::Replay) {
            if (options.replay.empty()) throw std::invalid_argument("replay mode requires --replay FILE");
            const lima::SolutionTrace trace = lima::SolutionTrace::load(options.replay, map);
#ifdef LIMA_HAS_SDL2
            return lima::run_replay(map, trace, {options.fps, options.max_steps});
#else
            throw std::runtime_error("this build has no SDL2 GUI support");
#endif
        }

        const auto solve_started = std::chrono::steady_clock::now();
        const std::uint64_t task_seed = options.seed.value_or(random_seed());
        std::vector<lima::Task> tasks = options.scenario.empty()
            ? lima::make_random_tasks(map, options.agents, task_seed)
            : lima::load_scenario(options.scenario, options.agents);
        if (tasks.empty()) throw std::runtime_error("no tasks were loaded");
        if (options.scenario.empty() && options.mode != RunMode::Debug) std::cout << "seed=" << task_seed << '\n';

        std::vector<std::vector<lima::Coord>> preset_routes;
        if (!options.routes.empty()) preset_routes = load_routes(options.routes);
        if (!options.goal_sequences.empty()) {
            if (options.sim.goal_behavior != lima::GoalBehavior::Lifelong)
                throw std::invalid_argument("--goal-sequences requires --goal-behavior lifelong");
            options.sim.lifelong_goal_sequences = load_routes(options.goal_sequences);
        }
        lima::Simulator simulator(std::move(map), tasks, options.planner, task_seed, options.sim, preset_routes);
        if (options.no_trace && !options.output.empty())
            throw std::invalid_argument("--no-trace and --output are mutually exclusive");
        std::filesystem::path output_path = options.output;
        if (options.mode == RunMode::Solve && output_path.empty() && !options.no_trace)
            output_path = "build/result.txt";
        std::unique_ptr<lima::SolutionTrace> recorder;
        if (!output_path.empty()) {
            recorder = std::make_unique<lima::SolutionTrace>(
                simulator.map(), simulator.agents(), options.map.string(), options.validate_conflicts,
                options.sim.goal_behavior == lima::GoalBehavior::Lifelong);
        }
        if (options.validate_conflicts && !recorder)
            throw std::invalid_argument("--validate-conflicts requires solve mode or --output FILE");
        if (options.mode == RunMode::Debug) {
            const int result = run_debug(simulator, options, recorder.get());
            if (recorder) {
                recorder->set_computation_time(std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - solve_started).count());
                recorder->save(output_path, simulator.map(), simulator.done());
            }
            return result;
        }
        if (options.mode == RunMode::Realtime) {
#ifdef LIMA_HAS_SDL2
            const int viewer_result = lima::run_viewer(simulator, {options.fps, options.max_steps}, recorder.get());
            if (recorder) recorder->save(output_path, simulator.map(), simulator.done());
            simulator.write_metrics();
            const auto& stats = simulator.stats();
            std::cout << "status=" << (simulator.done() ? "completed" : "closed") << " steps=" << stats.timestep
                      << " completed=" << stats.completed << '/' << tasks.size() << " moves=" << stats.committed_moves
                      << " waits=" << stats.waits << " deadlocks=" << stats.detected_deadlocks
                      << " intersections=" << simulator.topology().intersections().size() << '\n';
            return viewer_result;
#else
            throw std::runtime_error("this build has no SDL2 GUI support");
#endif
        }
        while (!simulator.done() && simulator.stats().timestep < options.max_steps) {
            if (!simulator.step()) break;
            if (recorder) recorder->append(simulator.agents());
        }
        const double solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_started).count();
        if (recorder) recorder->set_computation_time(solve_seconds);
        if (recorder) recorder->save(output_path, simulator.map(), simulator.done());
        simulator.write_metrics();
        const auto& stats = simulator.stats();
        std::cout << "status=" << (simulator.done() ? "completed" : "step_limit")
                  << " steps=" << stats.timestep
                  << " completed=" << stats.completed << '/' << tasks.size()
                  << " moves=" << stats.committed_moves
                  << " waits=" << stats.waits;
        if (options.sim.failure_probability != 0.0)
            std::cout << " failures=" << stats.command_failures;
        std::cout << " deadlocks=" << stats.detected_deadlocks
                  << " intersections=" << simulator.topology().intersections().size()
                  << " elapsed_seconds=" << solve_seconds;
        if (options.mode == RunMode::Solve) {
            if (recorder && recorder->validation_enabled()) {
                std::cout << " validation="
                          << (recorder->vertex_conflicts() == 0 && recorder->edge_conflicts() == 0 ? "ok" : "conflict")
                          << " vertex_conflicts=" << recorder->vertex_conflicts()
                          << " edge_conflicts=" << recorder->edge_conflicts();
            }
        }
        print_provenance(options);
        std::cout << '\n';
        return simulator.done() ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "lima: " << error.what() << '\n';
        return 1;
    }
}
