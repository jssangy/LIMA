#include "lima/planning/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace lima {
namespace {

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

std::unique_ptr<Planner> make_planner(const PlannerKind kind, const GridMap& map, std::mt19937_64& rng) {
    if (kind == PlannerKind::Bfs) return std::make_unique<BfsPlanner>(map, rng);
    if (kind == PlannerKind::AStar) return std::make_unique<AStarPlanner>(map);
    throw std::invalid_argument("unsupported planner");
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
