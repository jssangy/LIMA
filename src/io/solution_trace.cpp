#include "lima/io/solution_trace.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace lima {
namespace {

std::vector<Coord> parse_coordinates(const std::string_view line) {
    std::vector<Coord> result;
    std::size_t cursor = 0;
    while ((cursor = line.find('(', cursor)) != std::string_view::npos) {
        const std::size_t comma = line.find(',', cursor + 1);
        const std::size_t close = line.find(')', comma + 1);
        if (comma == std::string_view::npos || close == std::string_view::npos) {
            throw std::runtime_error("malformed coordinate list");
        }
        const int x = std::stoi(std::string(line.substr(cursor + 1, comma - cursor - 1)));
        const int y = std::stoi(std::string(line.substr(comma + 1, close - comma - 1)));
        result.push_back({x, y});
        cursor = close + 1;
    }
    return result;
}

std::vector<CellId> to_cells(const std::vector<Coord>& coordinates, const GridMap& map) {
    std::vector<CellId> result;
    result.reserve(coordinates.size());
    for (const Coord coordinate : coordinates) {
        if (!map.traversable(coordinate)) throw std::runtime_error("solution contains a non-traversable coordinate");
        result.push_back(map.cell(coordinate));
    }
    return result;
}

}  // namespace

SolutionTrace::SolutionTrace(const GridMap& map, const std::span<const Agent> agents, std::string map_file)
    : map_file_(std::move(map_file)) {
    starts_.reserve(agents.size());
    goals_.reserve(agents.size());
    for (const Agent& agent : agents) {
        if (!map.traversable(agent.position) || !map.traversable(agent.goal)) {
            throw std::logic_error("cannot record invalid agent endpoint");
        }
        starts_.push_back(agent.position);
        goals_.push_back(agent.goal);
    }
    append(agents);
}

void SolutionTrace::append(const std::span<const Agent> agents) {
    if (agents.size() != agent_count()) throw std::logic_error("solution agent count changed while recording");
    const bool has_previous = frame_count() > 0;
    const auto previous = has_previous ? frame(frame_count() - 1) : std::span<const CellId>{};
    std::vector<CellId> current;
    current.reserve(agents.size());
    for (const Agent& agent : agents) current.push_back(agent.position);

    // Temporary trace validation: completed robots are removed from simulator
    // occupancy, so include a robot in this transition when it was active at
    // either endpoint.  This still validates its final move into the goal.
    std::unordered_map<CellId, std::size_t> current_occupancy;
    current_occupancy.reserve(agents.size() * 2);
    for (std::size_t i = 0; i < agents.size(); ++i) {
        const bool participates = agents[i].active
            || (has_previous && i < previous_active_.size() && previous_active_[i]);
        if (!participates) continue;
        if (!current_occupancy.emplace(current[i], i).second) ++vertex_conflicts_;
    }
    if (has_previous) {
        std::unordered_map<CellId, std::size_t> previous_occupancy;
        previous_occupancy.reserve(agents.size() * 2);
        for (std::size_t i = 0; i < agents.size(); ++i)
            if (i < previous_active_.size() && previous_active_[i]) previous_occupancy.emplace(previous[i], i);
        for (std::size_t i = 0; i < agents.size(); ++i) {
            if (i >= previous_active_.size() || !previous_active_[i] || previous[i] == current[i]) continue;
            const auto occupant = previous_occupancy.find(current[i]);
            if (occupant == previous_occupancy.end()) continue;
            const std::size_t j = occupant->second;
            if (j > i && j < current.size() && previous[j] == current[i] && current[j] == previous[i])
                ++edge_conflicts_;
        }
    }

    configurations_.reserve(configurations_.size() + agents.size());
    configurations_.insert(configurations_.end(), current.begin(), current.end());
    previous_active_.clear();
    previous_active_.reserve(agents.size());
    for (const Agent& agent : agents) previous_active_.push_back(agent.active);
}

std::span<const CellId> SolutionTrace::frame(const std::size_t timestep) const {
    if (timestep >= frame_count()) throw std::out_of_range("solution timestep out of range");
    return {configurations_.data() + timestep * agent_count(), agent_count()};
}

void SolutionTrace::save(const std::filesystem::path& path, const GridMap& map, const bool solved) const {
    if (frame_count() == 0) throw std::runtime_error("cannot save an empty solution");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write solution: " + path.string());

    std::uint64_t sum_of_costs = 0;
    for (std::size_t agent = 0; agent < agent_count(); ++agent) {
        std::size_t cost = frame_count();
        while (cost > 0 && frame(cost - 1)[agent] == goals_[agent]) --cost;
        sum_of_costs += cost;
    }
    output << "agents=" << agent_count() << '\n';
    output << "map_file=" << std::filesystem::path(map_file_).filename().string() << '\n';
    output << "solver=lima\n";
    output << "solved=" << (solved ? 1 : 0) << '\n';
    output << "soc=" << sum_of_costs << '\n';
    output << "soc_lb=0\n";
    output << "makespan=" << frame_count() - 1 << '\n';
    output << "makespan_lb=0\n";
    output << "sum_of_loss=" << sum_of_costs << '\n';
    output << "sum_of_loss_lb=0\n";
    output << "comp_time=" << computation_time_seconds_ << '\n';
    output << "validation=" << (vertex_conflicts_ == 0 && edge_conflicts_ == 0 ? "ok" : "conflict") << '\n';
    output << "vertex_conflicts=" << vertex_conflicts_ << '\n';
    output << "edge_conflicts=" << edge_conflicts_ << '\n';
    output << "starts=";
    for (const CellId cell : starts_) {
        const Coord c = map.coord(cell);
        output << '(' << c.x << ',' << c.y << "),";
    }
    output << "\ngoals=";
    for (const CellId cell : goals_) {
        const Coord c = map.coord(cell);
        output << '(' << c.x << ',' << c.y << "),";
    }
    output << "\nsolution=\n";
    for (std::size_t timestep = 0; timestep < frame_count(); ++timestep) {
        output << timestep << ':';
        for (const CellId cell : frame(timestep)) {
            const Coord c = map.coord(cell);
            output << '(' << c.x << ',' << c.y << "),";
        }
        output << '\n';
    }
}

SolutionTrace SolutionTrace::load(const std::filesystem::path& path, const GridMap& map) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open solution: " + path.string());
    SolutionTrace trace;
    std::size_t declared_agents = 0;
    bool reading_solution = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.starts_with("agents=")) declared_agents = std::stoull(line.substr(7));
        else if (line.starts_with("map_file=")) trace.map_file_ = line.substr(9);
        else if (line.starts_with("solved=")) trace.solved_ = std::stoi(line.substr(7)) != 0;
        else if (line.starts_with("starts=")) trace.starts_ = to_cells(parse_coordinates(line), map);
        else if (line.starts_with("goals=")) trace.goals_ = to_cells(parse_coordinates(line), map);
        else if (line == "solution=") reading_solution = true;
        else if (reading_solution && !line.empty()) {
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) throw std::runtime_error("malformed solution frame");
            auto cells = to_cells(parse_coordinates(std::string_view(line).substr(colon + 1)), map);
            if (cells.size() != declared_agents) throw std::runtime_error("solution frame agent count mismatch");
            trace.configurations_.insert(trace.configurations_.end(), cells.begin(), cells.end());
        }
    }
    if (declared_agents == 0 || trace.starts_.size() != declared_agents || trace.goals_.size() != declared_agents
        || trace.frame_count() == 0) {
        throw std::runtime_error("incomplete solution file");
    }
    return trace;
}

}  // namespace lima
