#include "lima/planning/planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

namespace lima {
namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t guidance_seed(
    const std::uint64_t seed, const CellId low, const CellId high) {
    std::uint64_t state = splitmix64(seed ^ static_cast<std::uint64_t>(low));
    state = splitmix64(state ^ static_cast<std::uint64_t>(high));
    return splitmix64(state ^ 0x47554944ULL);
}

std::vector<CellId> reconstruct(const CellId start, const CellId goal, const std::vector<CellId>& parent) {
    if (start == goal) return {start};
    if (goal < 0 || static_cast<std::size_t>(goal) >= parent.size() || parent[static_cast<std::size_t>(goal)] < 0) return {};
    std::vector<CellId> path;
    for (CellId at = goal;; at = parent[static_cast<std::size_t>(at)]) {
        path.push_back(at);
        if (at == start) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::uint64_t directed_key(const CellId source, const CellId destination) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32U)
        | static_cast<std::uint32_t>(destination);
}

std::uint64_t state_key(const CellId cell, const std::int32_t steps) {
    return directed_key(cell, steps);
}

}  // namespace

std::vector<CellId> BfsPlanner::plan(const CellId start, const CellId goal) {
    if (!map_.traversable(start) || !map_.traversable(goal)) return {};
    // Highway reconnection and local recovery can produce many transient
    // waypoint goals.  Keeping every reverse field grows without bound on a
    // large map (one cross_3030 field is roughly 140 KiB).
    constexpr std::size_t kMaxCachedFields = 512;
    if (!fields_.contains(goal)) {
        if (fields_.size() >= kMaxCachedFields) {
            fields_.erase(field_order_.front());
            field_order_.pop_front();
        }
        field_order_.push_back(goal);
    }
    auto [it, inserted] = fields_.try_emplace(goal);
    auto& distance = it->second;
    if (inserted) {
        distance.assign(static_cast<std::size_t>(map_.cell_count()), -1);
        std::queue<CellId> queue;
        distance[static_cast<std::size_t>(goal)] = 0;
        queue.push(goal);
        while (!queue.empty()) {
            const CellId current = queue.front();
            queue.pop();
            for (const CellId next : map_.neighbors(current)) {
                if (distance[static_cast<std::size_t>(next)] >= 0) continue;
                distance[static_cast<std::size_t>(next)] = distance[static_cast<std::size_t>(current)] + 1;
                queue.push(next);
            }
        }
    }
    if (distance[static_cast<std::size_t>(start)] < 0) return {};

    std::vector<CellId> path{start};
    CellId current = start;
    while (current != goal) {
        std::vector<CellId> best;
        int best_distance = distance[static_cast<std::size_t>(current)];
        for (const CellId next : map_.neighbors(current)) {
            const int candidate = distance[static_cast<std::size_t>(next)];
            if (candidate >= 0 && candidate < best_distance) {
                best = {next};
                best_distance = candidate;
            } else if (candidate >= 0 && candidate == best_distance) {
                best.push_back(next);
            }
        }
        if (best.empty()) return {};
        std::uniform_int_distribution<std::size_t> choose(0, best.size() - 1);
        current = best[choose(rng_)];
        path.push_back(current);
    }
    return path;
}

std::vector<CellId> AStarPlanner::plan(const CellId start, const CellId goal) {
    if (!map_.traversable(start) || !map_.traversable(goal)) return {};
    struct Node {
        int f;
        int g;
        CellId cell;
        bool operator>(const Node& rhs) const noexcept {
            if (f != rhs.f) return f > rhs.f;
            if (g != rhs.g) return g < rhs.g;
            return cell > rhs.cell;
        }
    };
    const auto heuristic = [&](const CellId id) {
        const Coord a = map_.coord(id);
        const Coord b = map_.coord(goal);
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    };

    constexpr int inf = std::numeric_limits<int>::max();
    std::vector<int> g_score(static_cast<std::size_t>(map_.cell_count()), inf);
    std::vector<CellId> parent(static_cast<std::size_t>(map_.cell_count()), kInvalidCell);
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;
    g_score[static_cast<std::size_t>(start)] = 0;
    open.push({heuristic(start), 0, start});
    while (!open.empty()) {
        const Node node = open.top();
        open.pop();
        if (node.g != g_score[static_cast<std::size_t>(node.cell)]) continue;
        if (node.cell == goal) return reconstruct(start, goal, parent);
        for (const CellId next : map_.neighbors(node.cell)) {
            const int candidate = node.g + 1;
            if (candidate >= g_score[static_cast<std::size_t>(next)]) continue;
            g_score[static_cast<std::size_t>(next)] = candidate;
            parent[static_cast<std::size_t>(next)] = node.cell;
            open.push({candidate + heuristic(next), candidate, next});
        }
    }
    return {};
}

double StaticGuidancePlanner::edge_cost(
    const CellId source, const CellId destination) const {
    const CellId low = std::min(source, destination);
    const CellId high = std::max(source, destination);
    const bool preferred_low_to_high = (guidance_seed(seed_, low, high) & 1ULL) != 0;
    const bool low_to_high = source == low;
    return 1.0 + (preferred_low_to_high == low_to_high ? 0.0 : penalty_);
}

std::vector<CellId> StaticGuidancePlanner::plan(const CellId start, const CellId goal) {
    if (!map_.traversable(start) || !map_.traversable(goal)) return {};
    if (start == goal) return {start};
    constexpr std::size_t kMaxCachedFields = 512;
    if (!fields_.contains(goal)) {
        if (fields_.size() >= kMaxCachedFields) {
            fields_.erase(field_order_.front());
            field_order_.pop_front();
        }
        field_order_.push_back(goal);
    }
    auto [it, inserted] = fields_.try_emplace(goal);
    auto& distance = it->second;
    if (inserted) {
        constexpr double infinity = std::numeric_limits<double>::infinity();
        struct Node {
            double cost;
            CellId cell;
            bool operator>(const Node& rhs) const noexcept {
                if (cost != rhs.cost) return cost > rhs.cost;
                return cell > rhs.cell;
            }
        };
        distance.assign(static_cast<std::size_t>(map_.cell_count()), infinity);
        std::priority_queue<Node, std::vector<Node>, std::greater<>> open;
        distance[static_cast<std::size_t>(goal)] = 0.0;
        open.push({0.0, goal});
        while (!open.empty()) {
            const Node node = open.top();
            open.pop();
            if (node.cost > distance[static_cast<std::size_t>(node.cell)] + 1e-12) continue;
            for (const CellId predecessor : map_.neighbors(node.cell)) {
                const double candidate = node.cost + edge_cost(predecessor, node.cell);
                auto& value = distance[static_cast<std::size_t>(predecessor)];
                if (candidate + 1e-12 >= value) continue;
                value = candidate;
                open.push({candidate, predecessor});
            }
        }
    }
    if (!std::isfinite(distance[static_cast<std::size_t>(start)])) return {};

    std::vector<CellId> route{start};
    CellId current = start;
    while (current != goal) {
        CellId best = kInvalidCell;
        double best_cost = std::numeric_limits<double>::infinity();
        for (const CellId next : map_.neighbors(current)) {
            const double candidate = edge_cost(current, next)
                + distance[static_cast<std::size_t>(next)];
            if (candidate + 1e-12 < best_cost
                || (std::abs(candidate - best_cost) <= 1e-12 && next < best)) {
                best = next;
                best_cost = candidate;
            }
        }
        if (best == kInvalidCell
            || best_cost > distance[static_cast<std::size_t>(current)] + 1e-9) return {};
        current = best;
        route.push_back(current);
    }
    return route;
}

TrafficFlowPlanner::TrafficFlowPlanner(
    const GridMap& map, const double max_stretch, const double vertex_weight,
    const double edge_weight, const double contraflow_weight)
    : map_(map), max_stretch_(max_stretch), vertex_weight_(vertex_weight),
      edge_weight_(edge_weight), contraflow_weight_(contraflow_weight),
      vertex_load_(static_cast<std::size_t>(map.cell_count()), 0.0),
      edge_load_(static_cast<std::size_t>(map.cell_count()) * 4U, 0.0) {
    if (max_stretch < 1.0 || vertex_weight < 0.0 || edge_weight < 0.0
        || contraflow_weight < 0.0) {
        throw std::invalid_argument("invalid traffic-flow planner parameters");
    }
}

const std::vector<std::int32_t>& TrafficFlowPlanner::distance_field(
    const CellId goal) {
    constexpr std::size_t kMaxCachedFields = 128;
    if (!fields_.contains(goal)) {
        if (fields_.size() >= kMaxCachedFields) {
            fields_.erase(field_order_.front());
            field_order_.pop_front();
        }
        field_order_.push_back(goal);
    }
    auto [it, inserted] = fields_.try_emplace(goal);
    auto& distance = it->second;
    if (inserted) {
        distance.assign(static_cast<std::size_t>(map_.cell_count()), -1);
        std::queue<CellId> queue;
        distance[static_cast<std::size_t>(goal)] = 0;
        queue.push(goal);
        while (!queue.empty()) {
            const CellId current = queue.front();
            queue.pop();
            for (const CellId next : map_.neighbors(current)) {
                auto& value = distance[static_cast<std::size_t>(next)];
                if (value >= 0) continue;
                value = distance[static_cast<std::size_t>(current)] + 1;
                queue.push(next);
            }
        }
    }
    return distance;
}

std::size_t TrafficFlowPlanner::directed_index(
    const CellId source, const CellId destination) const {
    const Coord a = map_.coord(source);
    const Coord b = map_.coord(destination);
    std::size_t direction = 0;
    if (b.x == a.x && b.y + 1 == a.y) direction = 0;
    else if (b.x + 1 == a.x && b.y == a.y) direction = 1;
    else if (b.x == a.x + 1 && b.y == a.y) direction = 2;
    else if (b.x == a.x && b.y == a.y + 1) direction = 3;
    else throw std::logic_error("traffic-flow load update is not 4-connected");
    return static_cast<std::size_t>(source) * 4U + direction;
}

double TrafficFlowPlanner::directed_load(
    const CellId source, const CellId destination) const {
    return edge_load_[directed_index(source, destination)];
}

void TrafficFlowPlanner::note_assigned_route(
    const std::span<const CellId> route) {
    for (const CellId cell : route) {
        vertex_load_[static_cast<std::size_t>(cell)] += 1.0;
    }
    for (std::size_t i = 1; i < route.size(); ++i) {
        edge_load_[directed_index(route[i - 1], route[i])] += 1.0;
    }
}

std::vector<CellId> TrafficFlowPlanner::plan(
    const CellId start, const CellId goal) {
    if (!map_.traversable(start) || !map_.traversable(goal)) return {};
    if (start == goal) return {start};
    const auto& distance = distance_field(goal);
    const std::int32_t shortest = distance[static_cast<std::size_t>(start)];
    if (shortest < 0) return {};
    const std::int32_t maximum = std::max(
        shortest, static_cast<std::int32_t>(std::floor(
            static_cast<double>(shortest) * max_stretch_ + 1e-9)));

    const auto edge_cost_for = [&](const CellId source, const CellId destination) {
        return 1.0
            + vertex_weight_ * vertex_load_[static_cast<std::size_t>(destination)]
            + edge_weight_ * directed_load(source, destination)
            + contraflow_weight_ * directed_load(destination, source);
    };

    // A feasible shortest route supplies an upper bound. The unconstrained
    // flow optimum is exact whenever it already satisfies the stretch budget,
    // which is the common case.
    std::vector<CellId> incumbent_route{start};
    double incumbent_cost = 0.0;
    for (CellId current = start; current != goal;) {
        CellId best = kInvalidCell;
        double best_edge = std::numeric_limits<double>::infinity();
        const auto target = distance[static_cast<std::size_t>(current)] - 1;
        for (const CellId next : map_.neighbors(current)) {
            if (distance[static_cast<std::size_t>(next)] != target) continue;
            const double candidate = edge_cost_for(current, next);
            if (candidate + 1e-12 < best_edge
                || (std::abs(candidate - best_edge) <= 1e-12 && next < best)) {
                best = next;
                best_edge = candidate;
            }
        }
        if (best == kInvalidCell) return {};
        incumbent_cost += best_edge;
        current = best;
        incumbent_route.push_back(current);
    }

    struct FlowNode {
        double estimate;
        double cost;
        std::int32_t steps;
        CellId cell;
        bool operator>(const FlowNode& rhs) const noexcept {
            if (estimate != rhs.estimate) return estimate > rhs.estimate;
            if (cost != rhs.cost) return cost > rhs.cost;
            if (steps != rhs.steps) return steps > rhs.steps;
            return cell > rhs.cell;
        }
    };
    const std::size_t cells = static_cast<std::size_t>(map_.cell_count());
    std::vector<double> best_cost(cells, std::numeric_limits<double>::infinity());
    std::vector<std::int32_t> best_steps(cells, std::numeric_limits<std::int32_t>::max());
    std::vector<CellId> flow_parent(cells, kInvalidCell);
    std::priority_queue<FlowNode, std::vector<FlowNode>, std::greater<>> flow_open;
    best_cost[static_cast<std::size_t>(start)] = 0.0;
    best_steps[static_cast<std::size_t>(start)] = 0;
    flow_open.push({static_cast<double>(shortest), 0.0, 0, start});
    while (!flow_open.empty()) {
        const FlowNode node = flow_open.top();
        flow_open.pop();
        const auto index = static_cast<std::size_t>(node.cell);
        if (std::abs(node.cost - best_cost[index]) > 1e-12
            || node.steps != best_steps[index]) continue;
        if (node.cell == goal) {
            auto route = reconstruct(start, goal, flow_parent);
            if (!route.empty()
                && static_cast<std::int32_t>(route.size() - 1) <= maximum) {
                return route;
            }
            break;
        }
        for (const CellId next : map_.neighbors(node.cell)) {
            const double candidate = node.cost + edge_cost_for(node.cell, next);
            const auto next_steps = node.steps + 1;
            const auto next_index = static_cast<std::size_t>(next);
            if (candidate + 1e-12 > best_cost[next_index]
                || (std::abs(candidate - best_cost[next_index]) <= 1e-12
                    && next_steps >= best_steps[next_index])) continue;
            best_cost[next_index] = candidate;
            best_steps[next_index] = next_steps;
            flow_parent[next_index] = node.cell;
            const auto remaining = distance[next_index];
            flow_open.push({candidate + static_cast<double>(std::max(0, remaining)),
                            candidate, next_steps, next});
        }
    }

    struct Node {
        double estimate;
        double cost;
        std::int32_t steps;
        CellId cell;
        bool operator>(const Node& rhs) const noexcept {
            if (estimate != rhs.estimate) return estimate > rhs.estimate;
            if (cost != rhs.cost) return cost > rhs.cost;
            if (steps != rhs.steps) return steps > rhs.steps;
            return cell > rhs.cell;
        }
    };
    using Label = std::pair<std::int32_t, double>;
    constexpr std::size_t kMaxLabelsPerCell = 16;
    std::vector<std::vector<Label>> labels(
        static_cast<std::size_t>(map_.cell_count()));
    std::unordered_map<std::uint64_t, std::uint64_t> parent;
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;
    labels[static_cast<std::size_t>(start)].push_back({0, 0.0});
    open.push({static_cast<double>(shortest), 0.0, 0, start});
    std::uint64_t goal_state = 0;
    bool found = false;
    while (!open.empty()) {
        const Node node = open.top();
        open.pop();
        const auto& current_labels = labels[static_cast<std::size_t>(node.cell)];
        const bool current = std::any_of(
            current_labels.begin(), current_labels.end(), [&](const Label& label) {
                return label.first == node.steps
                    && std::abs(label.second - node.cost) <= 1e-12;
            });
        if (!current) continue;
        if (node.cell == goal) {
            goal_state = state_key(node.cell, node.steps);
            found = true;
            break;
        }
        for (const CellId next : map_.neighbors(node.cell)) {
            const std::int32_t next_steps = node.steps + 1;
            const std::int32_t remaining = distance[static_cast<std::size_t>(next)];
            if (remaining < 0 || next_steps + remaining > maximum) continue;
            const double edge_cost = edge_cost_for(node.cell, next);
            const double candidate = node.cost + edge_cost;
            if (candidate + static_cast<double>(remaining) > incumbent_cost + 1e-12) continue;
            auto& next_labels = labels[static_cast<std::size_t>(next)];
            if (std::any_of(next_labels.begin(), next_labels.end(),
                    [&](const Label& label) {
                        return label.first <= next_steps
                            && label.second <= candidate + 1e-12;
                    })) {
                continue;
            }
            next_labels.erase(std::remove_if(
                next_labels.begin(), next_labels.end(), [&](const Label& label) {
                    return next_steps <= label.first
                        && candidate <= label.second + 1e-12;
                }), next_labels.end());
            next_labels.push_back({next_steps, candidate});
            if (next_labels.size() > kMaxLabelsPerCell) {
                std::sort(next_labels.begin(), next_labels.end(),
                    [](const Label& lhs, const Label& rhs) {
                        if (lhs.first != rhs.first) return lhs.first < rhs.first;
                        return lhs.second < rhs.second;
                    });
                std::size_t redundant = 1;
                double least_deviation = std::numeric_limits<double>::infinity();
                for (std::size_t i = 1; i + 1 < next_labels.size(); ++i) {
                    const auto& left = next_labels[i - 1];
                    const auto& middle = next_labels[i];
                    const auto& right = next_labels[i + 1];
                    const double ratio = static_cast<double>(middle.first - left.first)
                        / static_cast<double>(right.first - left.first);
                    const double interpolated = left.second
                        + ratio * (right.second - left.second);
                    const double scale = std::max({1.0, std::abs(left.second),
                                                   std::abs(middle.second),
                                                   std::abs(right.second)});
                    const double deviation = std::abs(middle.second - interpolated) / scale;
                    if (deviation < least_deviation) {
                        least_deviation = deviation;
                        redundant = i;
                    }
                }
                next_labels.erase(next_labels.begin() + static_cast<std::ptrdiff_t>(redundant));
                if (std::none_of(next_labels.begin(), next_labels.end(), [&](const Label& label) {
                        return label.first == next_steps
                            && std::abs(label.second - candidate) <= 1e-12;
                    })) continue;
            }
            const auto next_state = state_key(next, next_steps);
            parent[next_state] = state_key(node.cell, node.steps);
            open.push({
                candidate + static_cast<double>(remaining), candidate,
                next_steps, next,
            });
        }
    }
    if (!found) return incumbent_route;

    const std::uint64_t start_state = state_key(start, 0);
    std::vector<CellId> route{goal};
    auto state = goal_state;
    while (state != start_state) {
        const auto it = parent.find(state);
        if (it == parent.end()) return {};
        state = it->second;
        route.push_back(static_cast<CellId>(state >> 32U));
    }
    std::reverse(route.begin(), route.end());
    return route;
}

std::unique_ptr<Planner> make_planner(const PlannerKind kind, const GridMap& map, std::mt19937_64& rng) {
    if (kind == PlannerKind::Bfs) return std::make_unique<BfsPlanner>(map, rng);
    if (kind == PlannerKind::AStar) return std::make_unique<AStarPlanner>(map);
    throw std::invalid_argument("unsupported planner");
}

std::unique_ptr<Planner> make_static_guidance_planner(
    const GridMap& map, const std::uint64_t seed, const double penalty) {
    if (penalty < 0.0) throw std::invalid_argument("guidance penalty must be non-negative");
    return std::make_unique<StaticGuidancePlanner>(map, seed, penalty);
}

std::unique_ptr<Planner> make_traffic_flow_planner(
    const GridMap& map, const double max_stretch, const double vertex_weight,
    const double edge_weight, const double contraflow_weight) {
    return std::make_unique<TrafficFlowPlanner>(
        map, max_stretch, vertex_weight, edge_weight, contraflow_weight);
}

std::optional<SuffixRepair> repair_to_reference_suffix(
    Planner& planner, const CellId current, const std::span<const CellId> reference_route,
    const std::size_t reference_cursor) {
    if (reference_route.empty() || reference_cursor >= reference_route.size()) return std::nullopt;
    const std::size_t rejoin_index = std::min(reference_cursor + 1, reference_route.size() - 1);
    const CellId rejoin = reference_route[rejoin_index];
    auto bridge = planner.plan(current, rejoin);
    if (bridge.empty() || bridge.front() != current || bridge.back() != rejoin) return std::nullopt;

    SuffixRepair repair;
    repair.rejoin = rejoin;
    repair.reference_rejoin_index = rejoin_index;
    repair.bridge_edges = bridge.size() - 1;
    repair.route.reserve(bridge.size() + reference_route.size() - rejoin_index - 1);
    repair.route = std::move(bridge);
    repair.route.insert(repair.route.end(),
        reference_route.begin() + static_cast<std::ptrdiff_t>(rejoin_index + 1),
        reference_route.end());
    return repair;
}

std::vector<CellId> plan_avoiding(const GridMap& map, const CellId start, const CellId goal,
                                  const std::span<const std::uint8_t> blocked) {
    if (blocked.size() != static_cast<std::size_t>(map.cell_count())) {
        throw std::invalid_argument("blocked-cell table size mismatch");
    }
    if (!map.traversable(start) || !map.traversable(goal)) return {};

    struct Node {
        int f;
        int g;
        CellId cell;
        bool operator>(const Node& rhs) const noexcept {
            if (f != rhs.f) return f > rhs.f;
            if (g != rhs.g) return g < rhs.g;
            return cell > rhs.cell;
        }
    };
    const Coord destination = map.coord(goal);
    const auto heuristic = [&](const CellId id) {
        const Coord c = map.coord(id);
        return std::abs(c.x - destination.x) + std::abs(c.y - destination.y);
    };
    constexpr int inf = std::numeric_limits<int>::max();
    std::vector<int> score(static_cast<std::size_t>(map.cell_count()), inf);
    std::vector<CellId> parent(static_cast<std::size_t>(map.cell_count()), kInvalidCell);
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;
    score[static_cast<std::size_t>(start)] = 0;
    open.push({heuristic(start), 0, start});
    while (!open.empty()) {
        const Node node = open.top();
        open.pop();
        if (node.g != score[static_cast<std::size_t>(node.cell)]) continue;
        if (node.cell == goal) return reconstruct(start, goal, parent);
        for (const CellId next : map.neighbors(node.cell)) {
            if (next != goal && next != start && blocked[static_cast<std::size_t>(next)] != 0) continue;
            const int candidate = node.g + 1;
            if (candidate >= score[static_cast<std::size_t>(next)]) continue;
            score[static_cast<std::size_t>(next)] = candidate;
            parent[static_cast<std::size_t>(next)] = node.cell;
            open.push({candidate + heuristic(next), candidate, next});
        }
    }
    return {};
}

}  // namespace lima
