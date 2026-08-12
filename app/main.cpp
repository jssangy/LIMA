#include "lima/core/grid_map.hpp"
#include "lima/io/scenario_loader.hpp"
#include "lima/io/solution_trace.hpp"
#include "lima/io/debug_trace.hpp"
#include "lima/simulation/simulator.hpp"
#include "lima/viewer/viewer.hpp"

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

enum class RunMode { Realtime, Solve, Replay, Debug };

struct Options {
    std::filesystem::path map{"data/maps/cross_1.map"};
    std::filesystem::path scenario;
    std::size_t agents{15};
    std::optional<std::uint64_t> seed;
    std::uint64_t max_steps{1000};
    lima::PlannerKind planner{lima::PlannerKind::Bfs};
    lima::WorkloadMode workload{lima::WorkloadMode::OneShot};
    bool validate_conflicts{};
    double fps{20.0};
    RunMode mode{RunMode::Realtime};
    std::filesystem::path output;
    std::filesystem::path replay;
    std::filesystem::path debug_dir;
};

void usage() {
    std::cout << "usage: lima [--map FILE] [--scenario FILE] [--agents N] [--planner bfs|astar]"
                 " [--workload oneshot|lifelong]"
                 " [--seed N] [--max-steps N] [--fps N] [--validate-conflicts]"
                 " [--mode realtime|solve|replay|debug] [--output FILE] [--replay FILE]"
                 " [--debug-dir DIR]\n";
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
        else if (arg == "--replay") options.replay = value();
        else if (arg == "--debug-dir") options.debug_dir = value();
        else if (arg == "--workload") {
            const auto name = value();
            if (name == "oneshot" || name == "one-shot")
                options.workload = lima::WorkloadMode::OneShot;
            else if (name == "lifelong") options.workload = lima::WorkloadMode::Lifelong;
            else throw std::invalid_argument("workload must be oneshot or lifelong");
        }
        else if (arg == "--mode") {
            const auto name = value();
            if (name == "realtime") options.mode = RunMode::Realtime;
            else if (name == "solve") options.mode = RunMode::Solve;
            else if (name == "replay") options.mode = RunMode::Replay;
            else if (name == "debug") options.mode = RunMode::Debug;
            else throw std::invalid_argument("mode must be realtime, solve, replay, or debug");
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        std::string command_line;
        for (int index = 0; index < argc; ++index) {
            if (index != 0) command_line += ' ';
            command_line += argv[index];
        }
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
        if (options.workload == lima::WorkloadMode::Lifelong && options.scenario.empty()
            && lima::make_goal_candidates(map).size() <= tasks.size()) {
            throw std::runtime_error(
                "lifelong random mode requires more goal candidates than agents");
        }
        if (options.scenario.empty()) std::cout << "seed=" << task_seed << '\n';

        lima::Simulator simulator(std::move(map), tasks, options.planner, task_seed, options.workload);
        std::filesystem::path output_path = options.output;
        if (options.mode == RunMode::Solve && output_path.empty()) output_path = "build/result.txt";
        std::filesystem::path debug_directory = options.debug_dir;
        if (options.mode == RunMode::Debug && debug_directory.empty()) {
            debug_directory = std::filesystem::path("results/debug") /
                (options.map.stem().string() + "_" + std::to_string(tasks.size())
                 + "_seed" + std::to_string(task_seed));
        }
        if (options.mode == RunMode::Debug && output_path.empty())
            output_path = debug_directory / "solution.txt";
        const bool validate_conflicts = options.validate_conflicts || options.mode == RunMode::Debug;
        std::unique_ptr<lima::SolutionTrace> recorder;
        if (!output_path.empty()) {
            recorder = std::make_unique<lima::SolutionTrace>(
                simulator.map(), simulator.agents(), options.map.string(), validate_conflicts,
                simulator.lifelong());
        }
        if (options.validate_conflicts && !recorder)
            throw std::invalid_argument("--validate-conflicts requires solve mode or --output FILE");
        std::unique_ptr<lima::DebugTrace> debugger;
        if (options.mode == RunMode::Debug) {
            debugger = std::make_unique<lima::DebugTrace>(
                debug_directory, simulator, options.map.string(), options.scenario.string(),
                command_line, task_seed, options.planner, options.max_steps);
            std::cout << "debug_dir=" << debug_directory.string() << '\n';
        }
        if (options.mode == RunMode::Realtime) {
#ifdef LIMA_HAS_SDL2
            const int viewer_result = lima::run_viewer(simulator, {options.fps, options.max_steps}, recorder.get());
            const bool horizon_reached = simulator.lifelong()
                && simulator.stats().timestep >= options.max_steps;
            if (recorder) recorder->save(
                output_path, simulator.map(), simulator.done() || horizon_reached);
            const auto& stats = simulator.stats();
            std::cout << "status=" << (simulator.done() ? "completed"
                          : horizon_reached ? "horizon_reached" : "closed")
                      << " steps=" << stats.timestep;
            if (simulator.lifelong())
                std::cout << " tasks_completed=" << stats.completed_tasks
                          << " throughput=" << (stats.timestep == 0 ? 0.0
                              : static_cast<double>(stats.completed_tasks) / stats.timestep)
                          << " avg_task_latency=" << (stats.completed_tasks == 0 ? 0.0
                              : static_cast<double>(stats.total_task_latency) / stats.completed_tasks);
            else std::cout << " completed=" << stats.completed << '/' << tasks.size();
            std::cout << " moves=" << stats.committed_moves
                      << " waits=" << stats.waits << " deadlocks=" << stats.detected_deadlocks
                      << " intersections=" << simulator.topology().intersections().size() << '\n';
            return viewer_result;
#else
            throw std::runtime_error("this build has no SDL2 GUI support");
#endif
        }
        bool stopped_early = false;
        while (!simulator.done() && simulator.stats().timestep < options.max_steps) {
            if (!simulator.step()) {
                stopped_early = true;
                break;
            }
            if (recorder) recorder->append(simulator.agents());
            if (debugger) debugger->append(simulator);
        }
        const double solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_started).count();
        const bool horizon_reached = simulator.lifelong()
            && simulator.stats().timestep >= options.max_steps && !stopped_early;
        const bool successful = simulator.done() || horizon_reached;
        if (recorder) recorder->set_computation_time(solve_seconds);
        if (recorder) recorder->save(output_path, simulator.map(), successful);
        const auto& stats = simulator.stats();
        const std::string_view status = simulator.done() ? "completed"
            : (horizon_reached ? "horizon_reached"
            : (stopped_early ? "stalled" : "step_limit"));
        if (debugger) debugger->finish(
            simulator, status, solve_seconds,
            recorder ? recorder->vertex_conflicts() : 0,
            recorder ? recorder->edge_conflicts() : 0);
        std::cout << "status=" << status
                  << " steps=" << stats.timestep
                  << (simulator.lifelong()
                      ? " tasks_completed=" + std::to_string(stats.completed_tasks)
                      : " completed=" + std::to_string(stats.completed) + '/' + std::to_string(tasks.size()))
                  << " moves=" << stats.committed_moves
                  << " waits=" << stats.waits
                  << " deadlocks=" << stats.detected_deadlocks
                  << " intersections=" << simulator.topology().intersections().size()
                  << " elapsed_seconds=" << solve_seconds;
        if (simulator.lifelong()) {
            std::cout << " throughput=" << (stats.timestep == 0 ? 0.0
                          : static_cast<double>(stats.completed_tasks) / stats.timestep)
                      << " avg_task_latency=" << (stats.completed_tasks == 0 ? 0.0
                          : static_cast<double>(stats.total_task_latency) / stats.completed_tasks);
        }
        if (options.mode == RunMode::Solve || options.mode == RunMode::Debug) {
            if (recorder && recorder->validation_enabled()) {
                std::cout << " validation="
                          << (recorder->vertex_conflicts() == 0 && recorder->edge_conflicts() == 0 ? "ok" : "conflict")
                          << " vertex_conflicts=" << recorder->vertex_conflicts()
                          << " edge_conflicts=" << recorder->edge_conflicts();
            }
        }
        std::cout << '\n';
        return successful ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "lima: " << error.what() << '\n';
        return 1;
    }
}
