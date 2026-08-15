// CBS baseline (Sharon et al. 2015) with disappear-at-target semantics
// (Stern et al. 2019).  Standalone experiment binary: reuses lima_core only
// for the map loader and the scenario loader; it shares no code with the
// simulator, gating, or solver paths.
//
// Semantics:
//  * An agent occupies its path cell at every timestep t in [0, T_i], where
//    T_i is its completion (arrival) time.  At the arrival timestep the agent
//    still occupies its goal cell (so two agents sharing a goal cell at the
//    same timestep is a vertex conflict); for t > T_i the agent has vanished
//    and occupies nothing.
//  * Vertex conflict: two active agents in the same cell at the same t.
//  * Edge conflict: two active agents swap cells between t-1 and t.
//  * Cost of a path is T_i (unit cost per move or wait); objective is the
//    sum of individual costs (SOC).
//
// Determinism: no RNG.  All priority queues use total orders (ties broken by
// depth, cell id, and node insertion id) and conflicts are scanned in fixed
// (timestep, agent-id) order, so repeated runs produce identical output.

#include "lima/core/grid_map.hpp"
#include "lima/io/scenario_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using lima::AgentId;
using lima::CellId;
using lima::GridMap;

using Clock = std::chrono::steady_clock;
using Path = std::vector<CellId>;  // positions at t = 0..T; T = completion time

constexpr CellId kNoCell = lima::kInvalidCell;

struct Constraint {
    AgentId agent{lima::kNoAgent};
    CellId from{kNoCell};  // kNoCell => vertex constraint; else edge from -> cell
    CellId cell{kNoCell};  // forbidden occupancy (vertex) or arrival cell (edge)
    int time{0};           // occupancy (vertex) or arrival (edge) timestep
};

struct Conflict {
    bool edge{false};
    AgentId a{lima::kNoAgent};
    AgentId b{lima::kNoAgent};
    CellId u{kNoCell};  // vertex: the cell; edge: a's source (= b's destination)
    CellId v{kNoCell};  // vertex: the cell; edge: a's destination (= b's source)
    int time{0};
};

struct CTNode {
    int parent{-1};
    Constraint constraint;  // unused for the root
    std::vector<std::shared_ptr<const Path>> paths;
    long long cost{0};      // sum of individual completion times (SOC)
    long long conflict_count{0};
    std::optional<Conflict> first_conflict;
};

struct OpenEntry {
    long long cost{0};
    long long conflicts{0};
    int id{0};
};

struct OpenOrder {
    bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const {
        if (lhs.cost != rhs.cost) return lhs.cost > rhs.cost;
        if (lhs.conflicts != rhs.conflicts) return lhs.conflicts > rhs.conflicts;
        return lhs.id > rhs.id;
    }
};

using OpenQueue = std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenOrder>;

// Constraints gathered for one agent, hashed for O(1) lookups in the low level.
struct ConstraintTables {
    std::unordered_set<std::int64_t> vertex;  // t * C + cell
    std::unordered_set<std::int64_t> edge;    // (t * C + from) * C + to
    int latest{-1};                           // max constrained timestep
};

// Backward BFS from the goal over the 4-connected grid; -1 = unreachable.
std::shared_ptr<const std::vector<int>> goal_distance_field(const GridMap& map, const CellId goal) {
    auto dist = std::make_shared<std::vector<int>>(static_cast<std::size_t>(map.cell_count()), -1);
    std::deque<CellId> frontier;
    (*dist)[static_cast<std::size_t>(goal)] = 0;
    frontier.push_back(goal);
    while (!frontier.empty()) {
        const CellId cur = frontier.front();
        frontier.pop_front();
        const int next_d = (*dist)[static_cast<std::size_t>(cur)] + 1;
        for (const CellId next : map.neighbors(cur)) {
            if ((*dist)[static_cast<std::size_t>(next)] >= 0) continue;
            (*dist)[static_cast<std::size_t>(next)] = next_d;
            frontier.push_back(next);
        }
    }
    return dist;
}

struct AStarEntry {
    int f{0};
    int t{0};
    CellId cell{kNoCell};
};

struct AStarOrder {
    bool operator()(const AStarEntry& lhs, const AStarEntry& rhs) const {
        if (lhs.f != rhs.f) return lhs.f > rhs.f;
        if (lhs.t != rhs.t) return lhs.t < rhs.t;  // prefer deeper (larger g)
        return lhs.cell > rhs.cell;
    }
};

// Space-time A* for one agent.  Once the current timestep is past every
// constraint the heuristic is exact and the static grid is conflict-free for
// this agent, so the remainder is completed by greedy descent on the distance
// field (first neighbor in the map's fixed order with dist == d-1).
std::optional<Path> low_level(const GridMap& map, const CellId start, const std::vector<int>& dist,
                              const ConstraintTables& tables, const Clock::time_point deadline,
                              bool& timed_out) {
    const auto cells = static_cast<std::int64_t>(map.cell_count());
    if (dist[static_cast<std::size_t>(start)] < 0) return std::nullopt;
    const auto vertex_key = [cells](const int t, const CellId c) {
        return static_cast<std::int64_t>(t) * cells + c;
    };
    const auto edge_key = [cells, &vertex_key](const int t, const CellId from, const CellId to) {
        return vertex_key(t, from) * cells + to;
    };
    if (tables.vertex.contains(vertex_key(0, start))) return std::nullopt;

    const auto greedy_tail = [&](Path& path) {
        CellId cur = path.back();
        while (dist[static_cast<std::size_t>(cur)] > 0) {
            const int want = dist[static_cast<std::size_t>(cur)] - 1;
            for (const CellId next : map.neighbors(cur)) {
                if (dist[static_cast<std::size_t>(next)] == want) {
                    cur = next;
                    break;
                }
            }
            path.push_back(cur);
        }
    };

    std::unordered_map<std::int64_t, std::int64_t> parent;  // state key -> predecessor key
    std::priority_queue<AStarEntry, std::vector<AStarEntry>, AStarOrder> open;
    parent.emplace(vertex_key(0, start), -1);
    open.push({dist[static_cast<std::size_t>(start)], 0, start});

    const auto reconstruct = [&](const int t, const CellId cell) {
        Path path(static_cast<std::size_t>(t) + 1, kNoCell);
        std::int64_t key = vertex_key(t, cell);
        for (int i = t; i >= 0; --i) {
            path[static_cast<std::size_t>(i)] = static_cast<CellId>(key % cells);
            key = parent.at(key);
        }
        return path;
    };

    std::uint64_t pops = 0;
    while (!open.empty()) {
        if ((++pops & 2047U) == 0 && Clock::now() >= deadline) {
            timed_out = true;
            return std::nullopt;
        }
        const AStarEntry cur = open.top();
        open.pop();
        if (dist[static_cast<std::size_t>(cur.cell)] == 0) return reconstruct(cur.t, cur.cell);
        if (cur.t >= tables.latest) {
            Path path = reconstruct(cur.t, cur.cell);
            greedy_tail(path);
            return path;
        }
        const int nt = cur.t + 1;
        const auto try_push = [&](const CellId next) {
            if (dist[static_cast<std::size_t>(next)] < 0) return;
            if (tables.vertex.contains(vertex_key(nt, next))) return;
            if (next != cur.cell && tables.edge.contains(edge_key(nt, cur.cell, next))) return;
            const std::int64_t key = vertex_key(nt, next);
            if (!parent.emplace(key, vertex_key(cur.t, cur.cell)).second) return;
            open.push({nt + dist[static_cast<std::size_t>(next)], nt, next});
        };
        for (const CellId next : map.neighbors(cur.cell)) try_push(next);
        try_push(cur.cell);  // wait
    }
    return std::nullopt;
}

// Scan all timesteps in ascending order; record the earliest conflict (vertex
// before edge within a timestep, then agent-id order) and count every
// pairwise collision for the best-first tie-break.
void scan_conflicts(CTNode& node) {
    node.conflict_count = 0;
    node.first_conflict.reset();
    int makespan = 0;
    for (const auto& path : node.paths) {
        makespan = std::max(makespan, static_cast<int>(path->size()) - 1);
    }
    std::unordered_map<CellId, AgentId> occupant;
    std::unordered_map<std::int64_t, AgentId> movers;  // (from, to) -> agent
    const auto n = static_cast<AgentId>(node.paths.size());
    std::int64_t cells = 0;
    for (const auto& path : node.paths) {
        for (const CellId c : *path) cells = std::max(cells, static_cast<std::int64_t>(c) + 1);
    }
    for (int t = 0; t <= makespan; ++t) {
        occupant.clear();
        for (AgentId a = 0; a < n; ++a) {
            const Path& path = *node.paths[static_cast<std::size_t>(a)];
            if (t >= static_cast<int>(path.size())) continue;  // vanished
            const CellId cell = path[static_cast<std::size_t>(t)];
            const auto [it, inserted] = occupant.emplace(cell, a);
            if (inserted) continue;
            ++node.conflict_count;
            if (!node.first_conflict) {
                node.first_conflict = Conflict{false, it->second, a, cell, cell, t};
            }
        }
        if (t == 0) continue;
        movers.clear();
        for (AgentId a = 0; a < n; ++a) {
            const Path& path = *node.paths[static_cast<std::size_t>(a)];
            if (t >= static_cast<int>(path.size())) continue;  // vanished before the move
            const CellId from = path[static_cast<std::size_t>(t) - 1];
            const CellId to = path[static_cast<std::size_t>(t)];
            if (from == to) continue;
            const auto reverse = movers.find(static_cast<std::int64_t>(to) * cells + from);
            if (reverse != movers.end()) {
                ++node.conflict_count;
                if (!node.first_conflict) {
                    node.first_conflict = Conflict{true, reverse->second, a, to, from, t};
                }
            }
            movers.emplace(static_cast<std::int64_t>(from) * cells + to, a);
        }
    }
}

struct Options {
    std::filesystem::path map;
    std::filesystem::path scenario;
    std::size_t agents{0};
    double time_limit{60.0};
};

// Write one spatial route per agent in the waypoint format consumed by
// app/main.cpp --routes.  CBS paths are space-time paths, so repeated cells
// represent waits; collapsing each consecutive run preserves both endpoints
// and leaves only 4-connected spatial moves.
bool dump_paths(const GridMap& map,
                const std::vector<std::shared_ptr<const Path>>& paths) {
    const char* const dump = std::getenv("CBS_DUMP");
    if (!dump || *dump == '\0') return false;

    std::vector<Path> spatial_paths;
    spatial_paths.reserve(paths.size());
    for (const auto& path_ptr : paths) {
        if (!path_ptr || path_ptr->empty()) return false;
        Path spatial;
        spatial.reserve(path_ptr->size());
        for (const CellId cell : *path_ptr) {
            if (!map.traversable(cell)) return false;
            if (spatial.empty() || spatial.back() != cell) spatial.push_back(cell);
        }
        for (std::size_t i = 1; i < spatial.size(); ++i) {
            const lima::Coord previous = map.coord(spatial[i - 1]);
            const lima::Coord current = map.coord(spatial[i]);
            const int distance = std::abs(previous.x - current.x)
                                 + std::abs(previous.y - current.y);
            if (distance != 1) return false;
        }
        spatial_paths.push_back(std::move(spatial));
    }

    std::ofstream out(dump);
    if (!out) return false;
    for (const Path& path : spatial_paths) {
        for (std::size_t i = 0; i < path.size(); ++i) {
            const lima::Coord c = map.coord(path[i]);
            out << (i ? " " : "") << c.x << ' ' << c.y;
        }
        out << '\n';
    }
    out.close();
    return out.good();
}

struct IncumbentDump {
    bool dumped{false};
    long long conflicts{0};
};

IncumbentDump dump_best_conflict_incumbent(
    OpenQueue open, const std::deque<CTNode>& nodes, const GridMap& map,
    const std::optional<int> current_node = std::nullopt) {
    std::optional<OpenEntry> best;
    const auto consider = [&](const int node_id) {
        const CTNode& node = nodes[static_cast<std::size_t>(node_id)];
        if (!node.first_conflict) return;
        const OpenEntry candidate{node.cost, node.conflict_count, node_id};
        if (!best || OpenOrder{}(*best, candidate)) best = candidate;
    };
    if (current_node) consider(*current_node);
    while (!open.empty()) {
        consider(open.top().id);
        open.pop();
    }
    if (!best) return {};
    const CTNode& node = nodes[static_cast<std::size_t>(best->id)];
    return {dump_paths(map, node.paths), node.conflict_count};
}

std::optional<Options> parse_args(const int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << arg << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--map") {
            const char* v = value();
            if (!v) return std::nullopt;
            opts.map = v;
        } else if (arg == "--scenario") {
            const char* v = value();
            if (!v) return std::nullopt;
            opts.scenario = v;
        } else if (arg == "--agents") {
            const char* v = value();
            if (!v) return std::nullopt;
            try {
                opts.agents = static_cast<std::size_t>(std::stoull(v));
            } catch (const std::exception&) {
                std::cerr << "invalid --agents value: " << v << "\n";
                return std::nullopt;
            }
        } else if (arg == "--time-limit") {
            const char* v = value();
            if (!v) return std::nullopt;
            try {
                opts.time_limit = std::stod(v);
            } catch (const std::exception&) {
                std::cerr << "invalid --time-limit value: " << v << "\n";
                return std::nullopt;
            }
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return std::nullopt;
        }
    }
    if (opts.map.empty() || opts.scenario.empty() || opts.agents == 0) {
        std::cerr << "usage: cbs_baseline --map FILE --scenario FILE --agents N [--time-limit SEC]\n";
        return std::nullopt;
    }
    return opts;
}

void report(const bool solved, const std::size_t agents, const long long makespan,
            const long long soc, const long long expansions, const Clock::time_point start,
            const IncumbentDump incumbent = {}) {
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("solved=%d agents=%zu makespan=%lld soc=%lld expansions=%lld elapsed_s=%.3f "
                "incumbent_dumped=%d incumbent_conflicts=%lld\n",
                solved ? 1 : 0, agents, makespan, soc, expansions, elapsed,
                incumbent.dumped ? 1 : 0, incumbent.conflicts);
}

}  // namespace

int main(const int argc, char** argv) {
    const Clock::time_point start_time = Clock::now();
    const auto opts = parse_args(argc, argv);
    if (!opts) return 2;
    const Clock::time_point deadline =
        start_time + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(opts->time_limit));

    GridMap map;
    std::vector<lima::Task> tasks;
    try {
        map = GridMap::load(opts->map);
        tasks = lima::load_scenario(opts->scenario, opts->agents);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    if (tasks.size() != opts->agents) {
        std::cerr << "scenario has only " << tasks.size() << " entries, requested "
                  << opts->agents << "\n";
        return 2;
    }
    const auto n = tasks.size();
    std::vector<CellId> starts(n);
    std::vector<std::shared_ptr<const std::vector<int>>> dist_fields(n);
    std::unordered_map<CellId, std::shared_ptr<const std::vector<int>>> dist_cache;
    for (std::size_t a = 0; a < n; ++a) {
        if (!map.traversable(tasks[a].start) || !map.traversable(tasks[a].goal)) {
            std::cerr << "agent " << a << " has a non-traversable start or goal\n";
            return 2;
        }
        starts[a] = map.cell(tasks[a].start);
        const CellId goal = map.cell(tasks[a].goal);
        auto cached = dist_cache.find(goal);
        if (cached == dist_cache.end()) {
            cached = dist_cache.emplace(goal, goal_distance_field(map, goal)).first;
        }
        dist_fields[a] = cached->second;
    }

    std::deque<CTNode> nodes;
    long long expansions = 0;

    // Root: each agent planned independently (empty constraint set).
    {
        CTNode root;
        root.paths.reserve(n);
        const ConstraintTables empty_tables;
        for (std::size_t a = 0; a < n; ++a) {
            bool timed_out = false;
            auto path = low_level(map, starts[a], *dist_fields[a], empty_tables, deadline, timed_out);
            if (timed_out) {
                report(false, n, 0, 0, expansions, start_time);
                return 0;
            }
            if (!path) {
                report(false, n, 0, 0, expansions, start_time);  // goal unreachable
                return 0;
            }
            root.cost += static_cast<long long>(path->size()) - 1;
            root.paths.push_back(std::make_shared<const Path>(std::move(*path)));
        }
        scan_conflicts(root);
        nodes.push_back(std::move(root));
    }

    OpenQueue open;
    open.push({nodes[0].cost, nodes[0].conflict_count, 0});

    while (!open.empty()) {
        if (Clock::now() >= deadline) {
            report(false, n, 0, 0, expansions, start_time,
                   dump_best_conflict_incumbent(open, nodes, map));
            return 0;
        }
        const OpenEntry entry = open.top();
        open.pop();
        const int node_id = entry.id;
        if (!nodes[static_cast<std::size_t>(node_id)].first_conflict) {
            long long makespan = 0;
            for (const auto& path : nodes[static_cast<std::size_t>(node_id)].paths) {
                makespan = std::max(makespan, static_cast<long long>(path->size()) - 1);
            }
            const IncumbentDump incumbent{
                dump_paths(map, nodes[static_cast<std::size_t>(node_id)].paths), 0};
            report(true, n, makespan, nodes[static_cast<std::size_t>(node_id)].cost,
                   expansions, start_time, incumbent);
            return 0;
        }
        ++expansions;
        const Conflict conflict = *nodes[static_cast<std::size_t>(node_id)].first_conflict;
        const std::array<Constraint, 2> splits{
            conflict.edge ? Constraint{conflict.a, conflict.u, conflict.v, conflict.time}
                          : Constraint{conflict.a, kNoCell, conflict.v, conflict.time},
            conflict.edge ? Constraint{conflict.b, conflict.v, conflict.u, conflict.time}
                          : Constraint{conflict.b, kNoCell, conflict.v, conflict.time},
        };
        for (const Constraint& constraint : splits) {
            // Gather this agent's constraints along the branch, plus the new one.
            ConstraintTables tables;
            const auto cells = static_cast<std::int64_t>(map.cell_count());
            const auto add = [&](const Constraint& c) {
                if (c.agent != constraint.agent) return;
                if (c.from == kNoCell) {
                    tables.vertex.insert(static_cast<std::int64_t>(c.time) * cells + c.cell);
                } else {
                    tables.edge.insert(
                        (static_cast<std::int64_t>(c.time) * cells + c.from) * cells + c.cell);
                }
                tables.latest = std::max(tables.latest, c.time);
            };
            add(constraint);
            for (int walk = node_id; walk > 0; walk = nodes[static_cast<std::size_t>(walk)].parent) {
                add(nodes[static_cast<std::size_t>(walk)].constraint);
            }

            const auto agent = static_cast<std::size_t>(constraint.agent);
            bool timed_out = false;
            auto path = low_level(map, starts[agent], *dist_fields[agent], tables, deadline, timed_out);
            if (timed_out) {
                report(false, n, 0, 0, expansions, start_time,
                       dump_best_conflict_incumbent(open, nodes, map, node_id));
                return 0;
            }
            if (!path) continue;  // infeasible child: prune

            CTNode child;
            child.parent = node_id;
            child.constraint = constraint;
            child.paths = nodes[static_cast<std::size_t>(node_id)].paths;
            child.cost = nodes[static_cast<std::size_t>(node_id)].cost
                         - (static_cast<long long>(child.paths[agent]->size()) - 1)
                         + (static_cast<long long>(path->size()) - 1);
            child.paths[agent] = std::make_shared<const Path>(std::move(*path));
            scan_conflicts(child);
            const int child_id = static_cast<int>(nodes.size());
            nodes.push_back(std::move(child));
            open.push({nodes.back().cost, nodes.back().conflict_count, child_id});
        }
    }

    // Open list exhausted: the instance is unsolvable under these semantics
    // (e.g., duplicate start cells).
    report(false, n, 0, 0, expansions, start_time);
    return 0;
}
