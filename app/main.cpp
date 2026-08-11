#include "lima/core/grid_map.hpp"
#include "lima/io/scenario_loader.hpp"
#include "lima/io/solution_trace.hpp"
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

enum class RunMode { Realtime, Solve, Replay };

struct Options {
    std::filesystem::path map{"maps/cross_1.map"};
    std::filesystem::path scenario;
    std::size_t agents{15};
    std::optional<std::uint64_t> seed;
    std::uint64_t max_steps{1000};
    lima::PlannerKind planner{lima::PlannerKind::Bfs};
    bool gui{};
    double fps{20.0};
    RunMode mode{RunMode::Realtime};
    std::filesystem::path output;
    std::filesystem::path replay;
};

void usage() {
    std::cout << "usage: lima [--map FILE] [--scenario FILE] [--agents N] [--planner bfs|astar]"
                 " [--seed N] [--max-steps N] [--gui] [--fps N]"
                 " [--mode realtime|solve|replay] [--output FILE] [--replay FILE]\n";
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
        else if (arg == "--gui") options.gui = true;
        else if (arg == "--output") options.output = value();
        else if (arg == "--replay") options.replay = value();
        else if (arg == "--mode") {
            const auto name = value();
            if (name == "realtime") options.mode = RunMode::Realtime;
            else if (name == "solve") options.mode = RunMode::Solve;
            else if (name == "replay") options.mode = RunMode::Replay;
            else throw std::invalid_argument("mode must be realtime, solve, or replay");
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
        lima::GridMap map = lima::GridMap::load(options.map);
        if (options.fps <= 0.0) throw std::invalid_argument("fps must be greater than zero");

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
        if (options.scenario.empty()) std::cout << "seed=" << task_seed << '\n';

        lima::Simulator simulator(std::move(map), tasks, options.planner);
        std::filesystem::path output_path = options.output;
        if (options.mode == RunMode::Solve && output_path.empty()) output_path = "build/result.txt";
        std::unique_ptr<lima::SolutionTrace> recorder;
        if (!output_path.empty()) {
            recorder = std::make_unique<lima::SolutionTrace>(simulator.map(), simulator.agents(), options.map.string());
        }
        if (options.mode == RunMode::Solve && options.gui) {
            throw std::invalid_argument("solve mode is headless; use realtime mode with --gui --output to display and record");
        }
        if (options.gui) {
#ifdef LIMA_HAS_SDL2
            const int viewer_result = lima::run_viewer(simulator, {options.fps, options.max_steps}, recorder.get());
            if (recorder) recorder->save(output_path, simulator.map(), simulator.done());
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
        const auto& stats = simulator.stats();
        std::cout << "status=" << (simulator.done() ? "completed" : "step_limit")
                  << " steps=" << stats.timestep
                  << " completed=" << stats.completed << '/' << tasks.size()
                  << " moves=" << stats.committed_moves
                  << " waits=" << stats.waits
                  << " deadlocks=" << stats.detected_deadlocks
                  << " intersections=" << simulator.topology().intersections().size()
                  << " elapsed_seconds=" << solve_seconds;
        if (options.mode == RunMode::Solve) {
            if (recorder) {
                std::cout << " validation="
                          << (recorder->vertex_conflicts() == 0 && recorder->edge_conflicts() == 0 ? "ok" : "conflict")
                          << " vertex_conflicts=" << recorder->vertex_conflicts()
                          << " edge_conflicts=" << recorder->edge_conflicts();
            }
        }
        std::cout << '\n';
        return simulator.done() ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "lima: " << error.what() << '\n';
        return 1;
    }
}
