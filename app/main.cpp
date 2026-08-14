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
    lima::SimulatorConfig sim;
    lima::bench::BenchOptions bench;
};

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
    std::cout << "usage: lima [--map FILE] [--scenario FILE] [--agents N] [--planner bfs|astar]"
                 " [--seed N] [--max-steps N] [--fps N] [--validate-conflicts]"
                 " [--mode realtime|solve|replay|debug] [--output FILE|--no-trace] [--replay FILE]\n"
                 "            [--solver ida|greedy|beam|hybrid] [--solver-iterations N] [--bound-step N] [--no-fastpath]\n"
                 "            [--lb-mode legacy|bf|tt] [--dominance] [--solver-nodes N]\n"
                 "            [--routing dor|direct] [--capacity-formula code|paper] [--isolation-cap N]\n"
                 "            [--no-pibt-corridor] [--pibt-sink-yield] [--pibt-arm-retreat[-last]]\n"
                 "            [--pibt-age-rate] [--pibt-replan N] [--shuffle-order SEED] [--failure-prob P]\n"
                 "            [--no-discharge] [--metrics DIR] [--trace-jsonl FILE]\n"
                 "debug mode reads commands from stdin and answers in JSON:\n"
                 "            step [n] | state | agent ID | intersection ID | summary | invariants | quit\n";
}

Options parse(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> std::string_view {
            if (++i >= argc) throw std::invalid_argument("missing value after " + std::string(arg));
            return argv[i];
        };
        if (arg == "--map") options.map = value();
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
        else if (arg == "--solver") options.sim.solver.kind = std::string(value());
        else if (arg == "--solver-iterations") options.sim.solver.max_iterations = std::stoull(std::string(value()));
        else if (arg == "--bound-step") options.sim.solver.bound_step = std::stoi(std::string(value()));
        else if (arg == "--no-fastpath") options.sim.solver.greedy_fastpath = false;
        else if (arg == "--solver-max-capacity") options.sim.solver.max_capacity = std::stoi(std::string(value()));
        else if (arg == "--lb-mode") options.sim.solver.lb_mode = std::string(value());
        else if (arg == "--dominance") options.sim.solver.dominance = true;
        else if (arg == "--solver-nodes") options.sim.solver.max_nodes = std::stoull(std::string(value()));
        else if (arg == "--isolation-cap") options.sim.isolation.cap = std::stoi(std::string(value()));
        else if (arg == "--isolation-margin") options.sim.isolation.margin = std::stoi(std::string(value()));
        else if (arg == "--stall-threshold") options.sim.stall_threshold = static_cast<std::uint32_t>(std::stoul(std::string(value())));
        else if (arg == "--discharge-all-arms") options.sim.discharge.all_arms = true;
        else if (arg == "--discharge-stalled-neighbor") options.sim.discharge.allow_stalled_neighbor = true;
        else if (arg == "--discharge-deterministic") options.sim.discharge.deterministic_cycle = true;
        else if (arg == "--discharge-avail-weighted") options.sim.discharge.avail_weighted = true;
        else if (arg == "--discharge-random") {
            options.sim.discharge.deterministic_cycle = false;
            options.sim.discharge.avail_weighted = false;
        }
        else if (arg == "--discharge-partial") options.sim.discharge.partial_stall = std::stod(std::string(value()));
        else if (arg == "--admit-hysteresis") options.sim.isolation.hysteresis = std::stoi(std::string(value()));
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
            if (name == "dor") options.sim.direct_routing = false;
            else if (name == "direct") options.sim.direct_routing = true;
            else throw std::invalid_argument("routing must be dor or direct");
        }
        else if (arg == "--capacity-formula") {
            const auto name = value();
            if (name == "code") options.sim.isolation.formula = lima::CapacityFormula::SumMinusMax;
            else if (name == "paper") options.sim.isolation.formula = lima::CapacityFormula::SumMinusMaxPlusOne;
            else throw std::invalid_argument("capacity-formula must be code or paper");
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

bool has_non_default_config(const Options& options) {
    const lima::SolverConfig defaults;
    return options.sim.solver.kind != defaults.kind
        || options.sim.solver.max_iterations != defaults.max_iterations
        || options.sim.solver.bound_step != defaults.bound_step
        || options.sim.solver.greedy_fastpath != defaults.greedy_fastpath
        || options.sim.solver.max_capacity != defaults.max_capacity
        || options.sim.solver.lb_mode != defaults.lb_mode
        || options.sim.solver.dominance != defaults.dominance
        || options.sim.solver.max_nodes != defaults.max_nodes
        || options.sim.isolation.formula != lima::CapacityFormula::SumMinusMax
        || options.sim.isolation.cap >= 0
        || options.sim.isolation.margin != 0
        || options.sim.stall_threshold != 10
        || options.sim.discharge.all_arms
        || options.sim.discharge.allow_stalled_neighbor
        || !options.sim.discharge.deterministic_cycle
        || !options.sim.discharge.avail_weighted
        || options.sim.discharge.partial_stall != 1.0
        || options.sim.isolation.hysteresis != 0
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
    std::cout << " solver=" << options.sim.solver.kind
              << " routing=" << (options.sim.direct_routing ? "direct" : "dor")
              << " capacity=" << (options.sim.isolation.formula == lima::CapacityFormula::SumMinusMax ? "code" : "paper");
    if (options.sim.solver.lb_mode != "legacy") std::cout << " lb=" << options.sim.solver.lb_mode;
    if (options.sim.solver.dominance) std::cout << " dom=on";
    if (options.sim.isolation.cap >= 0) std::cout << " cap=" << options.sim.isolation.cap;
    if (!options.sim.pibt_corridor) std::cout << " pibt=off";
    if (options.sim.pibt_sink_yield) std::cout << " sink_yield=on";
    if (options.sim.pibt_arm_retreat) std::cout << " arm_retreat=on";
    if (options.sim.pibt_arm_retreat_last) std::cout << " arm_retreat=last";
    if (options.sim.pibt_age_rate) std::cout << " age_rate=on";
    if (options.sim.pibt_replan != 0) std::cout << " replan=" << options.sim.pibt_replan;
    if (options.sim.shuffle_order >= 0) std::cout << " shuffle=" << options.sim.shuffle_order;
    if (options.sim.failure_probability != 0.0)
        std::cout << " failure_prob=" << options.sim.failure_probability;
    if (!options.sim.discharge_enabled) std::cout << " discharge=off";
    if (options.no_trace) std::cout << " trace=off";
    std::cout << " commit=" << LIMA_COMMIT;
}

constexpr std::array<std::string_view, 8> kWaitReasonNames{
    "none", "scheduled_hold", "intersection_reserved", "intersection_capacity",
    "vertex_conflict", "edge_swap", "dependency", "schedule_group"};

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
