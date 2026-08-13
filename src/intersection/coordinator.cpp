#include "lima/intersection/coordinator.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace lima {
namespace {

using Lane = std::vector<std::optional<AgentId>>;

bool adjacent_or_equal(const CellId a, const CellId b, const Intersection& intersection) {
    if (a == b) return true;
    if (a == intersection.center || b == intersection.center) {
        const CellId other = a == intersection.center ? b : a;
        for (const auto& arm : intersection.arms) if (!arm.empty() && arm.front() == other) return true;
    }
    for (const auto& arm : intersection.arms) {
        for (std::size_t i = 0; i + 1 < arm.size(); ++i) {
            if ((arm[i] == a && arm[i + 1] == b) || (arm[i] == b && arm[i + 1] == a)) return true;
        }
    }
    return false;
}

std::optional<std::vector<CellId>> make_trimmed_egress(
    const Intersection& intersection, const IntersectionIntent& intent,
    const std::vector<CellId>& full_path) {
    if (full_path.empty()) return std::nullopt;
    std::size_t last_center = full_path.size();
    for (std::size_t i = full_path.size(); i-- > 0;) {
        if (full_path[i] == intersection.center) {
            last_center = i;
            break;
        }
    }
    if (last_center == full_path.size() || last_center + 1 >= full_path.size()) return std::nullopt;
    if (intersection.direction_of(full_path[last_center + 1]) != intent.exit) return std::nullopt;

    std::size_t farthest = last_center;
    std::int64_t farthest_distance = -1;
    for (std::size_t i = last_center + 1; i < full_path.size(); ++i) {
        const auto direction = static_cast<std::size_t>(intent.exit);
        if (direction >= 4) return std::nullopt;
        const auto& arm = intersection.arms[direction];
        const auto found = std::find(arm.begin(), arm.end(), full_path[i]);
        if (found == arm.end()) continue;
        const auto distance = static_cast<std::int64_t>(found - arm.begin());
        if (distance > farthest_distance) {
            farthest_distance = distance;
            farthest = i;
        }
    }
    if (farthest_distance < 0 || farthest + 1 >= full_path.size()) return std::nullopt;
    return std::vector<CellId>(full_path.begin(),
        full_path.begin() + static_cast<std::ptrdiff_t>(farthest + 1));
}

// True when the intent's next route step continues outward along its own arm
// (or holds position), i.e. the classic egress an unscheduled release cannot
// break.  `limit` is the usable arm depth (arm_limits), so the effective tip
// is arm[limit - 1] rather than the physical arm end under subset scheduling.
bool follows_natural_egress(const Intersection& intersection, const IntersectionIntent& intent,
                            const std::size_t limit) {
    const auto direction = static_cast<std::size_t>(intent.current);
    if (direction >= 4 || intent.current != intent.exit) return false;

    const auto& arm = intersection.arms[direction];
    const auto effective_end = arm.begin()
        + static_cast<std::ptrdiff_t>(std::min(limit, arm.size()));
    const auto current = std::find(arm.begin(), effective_end, intent.position);
    if (current == effective_end) return false;
    if (intent.next == intent.position) return true;

    const auto next = std::find(arm.begin(), arm.end(), intent.next);
    if (next != arm.end()) return next > current;

    // From the arm tip, a next cell outside this intersection is the normal
    // discharge step. Any other unclassified move is kept under scheduling.
    return current + 1 == effective_end && intersection.direction_of(intent.next) == Direction::None;
}

// True when any scheduled path moves inward (toward the center) along the
// given arm; releasing an egress AMR on that arm could then edge-swap with
// the active schedule.
bool has_inward_schedule_motion(
    const Intersection& intersection, const Direction direction,
    const std::unordered_map<AgentId, std::vector<CellId>>& paths) {
    const auto d = static_cast<std::size_t>(direction);
    if (d >= 4) return true;
    const auto& arm = intersection.arms[d];
    for (const auto& [id, path] : paths) {
        (void)id;
        for (std::size_t i = 0; i + 1 < path.size(); ++i) {
            const auto from = std::find(arm.begin(), arm.end(), path[i]);
            if (from == arm.end()) continue;
            if (path[i + 1] == intersection.center) return true;
            const auto to = std::find(arm.begin(), arm.end(), path[i + 1]);
            if (to != arm.end() && to < from) return true;
        }
    }
    return false;
}

}  // namespace

std::optional<std::vector<ScheduledPath>> IntersectionCoordinator::schedule(
    const Intersection& intersection, const std::span<const IntersectionIntent> intents,
    const std::array<int, 4>& stack_quotas, const std::array<int, 4>& arm_limits,
    ScheduleTelemetry* const telemetry) const {
    std::vector<Direction> directions;
    for (std::size_t d = 0; d < 4; ++d) {
        if (!intersection.arms[d].empty() && arm_limits[d] > 0) directions.push_back(static_cast<Direction>(d));
    }
    if (directions.size() < 3 || intents.size() < 2) return std::nullopt;
    const auto limit_of = [&](const std::size_t d) {
        return std::min(static_cast<std::size_t>(arm_limits[d]), intersection.arms[d].size());
    };

    std::unordered_map<AgentId, Direction> desired_direction;
    std::array<Lane, 4> current;
    for (const Direction d : directions) current[static_cast<std::size_t>(d)].resize(limit_of(static_cast<std::size_t>(d)));
    std::optional<AgentId> center_agent;
    for (const auto& intent : intents) {
        desired_direction[intent.agent] = intent.exit;
        if (intent.current == Direction::Center) {
            if (center_agent) return std::nullopt;
            center_agent = intent.agent;
            continue;
        }
        const auto d = static_cast<std::size_t>(intent.current);
        if (d >= 4) return std::nullopt;
        const auto& cells = intersection.arms[d];
        const auto found = std::find(cells.begin(), cells.end(), intent.position);
        if (found == cells.end()) return std::nullopt;
        const auto index = static_cast<std::size_t>(found - cells.begin());
        if (index >= limit_of(d)) return std::nullopt;  // participant beyond the usable depth
        current[d][index] = intent.agent;
    }

    std::vector<const IntersectionIntent*> participants;
    participants.reserve(intents.size());
    for (const auto& intent : intents) participants.push_back(&intent);

    std::array<Lane, 4> target;
    for (const Direction direction : directions) {
        const auto d = static_cast<std::size_t>(direction);
        target[d].resize(current[d].size());
        std::size_t write = 0;
        for (const auto id : current[d]) if (id) target[d][write++] = id;
    }
    if (center_agent) {
        Direction host = desired_direction[*center_agent];
        auto has_space = [&](const Direction d) {
            if (static_cast<std::size_t>(d) >= 4 || target[static_cast<std::size_t>(d)].empty()) return false;
            return !target[static_cast<std::size_t>(d)].back().has_value();
        };
        if (!has_space(host)) {
            std::size_t best_count = std::numeric_limits<std::size_t>::max();
            host = Direction::None;
            for (const Direction d : directions) {
                const auto count = static_cast<std::size_t>(std::count_if(target[static_cast<std::size_t>(d)].begin(),
                    target[static_cast<std::size_t>(d)].end(), [](const auto id) { return id.has_value(); }));
                if (count < target[static_cast<std::size_t>(d)].size() && count < best_count) {
                    best_count = count;
                    host = d;
                }
            }
        }
        if (!has_space(host)) return std::nullopt;
        auto& lane = target[static_cast<std::size_t>(host)];
        for (std::size_t i = lane.size() - 1; i > 0; --i) lane[i] = lane[i - 1];
        lane[0] = center_agent;
    }

    std::unordered_map<AgentId, std::vector<CellId>> paths;
    std::unordered_map<AgentId, std::pair<Direction, std::size_t>> current_location;
    std::unordered_map<AgentId, std::pair<Direction, std::size_t>> target_location;
    for (const IntersectionIntent* intent : participants) paths[intent->agent] = {intent->position};
    for (const Direction direction : directions) {
        const auto d = static_cast<std::size_t>(direction);
        for (std::size_t i = 0; i < current[d].size(); ++i) if (current[d][i]) current_location[*current[d][i]] = {direction, i};
        for (std::size_t i = 0; i < target[d].size(); ++i) if (target[d][i]) target_location[*target[d][i]] = {direction, i};
    }

    std::size_t prestage_steps = center_agent ? 1 : 0;
    for (const auto& [id, source] : current_location) {
        const auto destination = target_location.at(id);
        if (source.first != destination.first) return std::nullopt;
        const auto distance = source.second > destination.second ? source.second - destination.second : destination.second - source.second;
        prestage_steps = std::max(prestage_steps, distance);
    }
    for (std::size_t step = 1; step <= prestage_steps; ++step) {
        for (const IntersectionIntent* intent : participants) {
            CellId position = paths[intent->agent].back();
            if (center_agent && intent->agent == *center_agent) {
                const auto [direction, index] = target_location.at(intent->agent);
                (void)index;
                if (step == 1) position = intersection.arms[static_cast<std::size_t>(direction)][0];
            } else {
                const auto [direction, from] = current_location.at(intent->agent);
                const auto to = target_location.at(intent->agent).second;
                std::size_t index = from;
                if (from < to) index = std::min(to, from + step);
                else if (from > to) index = from - std::min(from - to, step);
                position = intersection.arms[static_cast<std::size_t>(direction)][index];
            }
            paths[intent->agent].push_back(position);
        }
    }

    std::unordered_map<Direction, int> direction_index;
    for (std::size_t i = 0; i < directions.size(); ++i) direction_index[directions[i]] = static_cast<int>(i);
    std::vector<std::vector<AgentId>> stacks(directions.size());
    std::vector<std::vector<int>> solver_stacks(directions.size());
    std::vector<int> capacities;
    for (std::size_t i = 0; i < directions.size(); ++i) {
        const auto d = static_cast<std::size_t>(directions[i]);
        capacities.push_back(static_cast<int>(target[d].size()));
        for (auto it = target[d].rbegin(); it != target[d].rend(); ++it) if (*it) {
            stacks[i].push_back(**it);
            Direction desired = desired_direction[**it];
            if (!direction_index.contains(desired)) desired = directions[i];
            solver_stacks[i].push_back(direction_index[desired]);
        }
    }

    SolverStats solver_stats;
    auto actions_opt = solver_.solve({solver_stacks, capacities}, &solver_stats);
    if (telemetry) {
        telemetry->intents = intents.size();
        telemetry->solver = solver_stats;
    }
    if (!actions_opt) return std::nullopt;
    std::vector<StackMove> actions = std::move(*actions_opt);

    // Apply neighbor-capacity quotas as a lightweight post-process. Overflow
    // robots may be parked on another arm.
    auto final_types = solver_stacks;
    for (const auto& [src, dst] : actions) {
        if (final_types[src].empty() || final_types[dst].size() >= static_cast<std::size_t>(capacities[dst])) return std::nullopt;
        const int item = final_types[src].back();
        final_types[src].pop_back();
        final_types[dst].push_back(item);
    }
    std::vector<int> quotas;
    quotas.reserve(directions.size());
    for (std::size_t i = 0; i < directions.size(); ++i) {
        const int quota = stack_quotas[static_cast<std::size_t>(directions[i])];
        quotas.push_back(std::clamp(quota, 0, capacities[i]));
    }
    for (std::size_t guard = 0; guard < 4096; ++guard) {
        int src = -1;
        int largest_overflow = 0;
        for (std::size_t i = 0; i < final_types.size(); ++i) {
            const int overflow = static_cast<int>(final_types[i].size()) - quotas[i];
            if (overflow > largest_overflow) {
                largest_overflow = overflow;
                src = static_cast<int>(i);
            }
        }
        if (src < 0) break;
        int dst = -1;
        int best_slack = 0;
        for (std::size_t i = 0; i < final_types.size(); ++i) {
            if (static_cast<int>(i) == src || final_types[i].size() >= static_cast<std::size_t>(capacities[i])) continue;
            const int slack = quotas[i] - static_cast<int>(final_types[i].size());
            if (slack > best_slack) {
                best_slack = slack;
                dst = static_cast<int>(i);
            }
        }
        if (dst < 0 || final_types[src].empty()) return std::nullopt;
        const int item = final_types[src].back();
        final_types[src].pop_back();
        final_types[dst].push_back(item);
        actions.emplace_back(src, dst);
    }

    const auto append_snapshot = [&](const std::optional<AgentId> center) {
        std::unordered_map<AgentId, CellId> positions;
        if (center) positions[*center] = intersection.center;
        for (std::size_t i = 0; i < directions.size(); ++i) {
            const auto& cells = intersection.arms[static_cast<std::size_t>(directions[i])];
            for (std::size_t near = 0; near < stacks[i].size(); ++near) positions[stacks[i][stacks[i].size() - 1 - near]] = cells[near];
        }
        for (const IntersectionIntent* intent : participants) {
            const auto found = positions.find(intent->agent);
            if (found == positions.end()) throw std::logic_error("scheduled agent vanished from stack state");
            paths[intent->agent].push_back(found->second);
        }
    };

    std::optional<AgentId> center;
    int pending_destination = -1;
    for (const auto& [src, dst] : actions) {
        if (src < 0 || dst < 0 || static_cast<std::size_t>(src) >= stacks.size()
            || static_cast<std::size_t>(dst) >= stacks.size()) return std::nullopt;

        // Finish the previous center push and pull the next source in the same
        // logical frame.
        if (center) {
            if (pending_destination < 0
                || stacks[static_cast<std::size_t>(pending_destination)].size()
                    >= static_cast<std::size_t>(capacities[static_cast<std::size_t>(pending_destination)])) return std::nullopt;
            stacks[static_cast<std::size_t>(pending_destination)].push_back(*center);
            center.reset();
        }
        if (stacks[src].empty()) return std::nullopt;
        const AgentId moving = stacks[src].back();
        stacks[src].pop_back();
        center = moving;
        pending_destination = dst;
        append_snapshot(center);
    }
    if (center) {
        if (stacks[static_cast<std::size_t>(pending_destination)].size()
            >= static_cast<std::size_t>(capacities[static_cast<std::size_t>(pending_destination)])) return std::nullopt;
        stacks[static_cast<std::size_t>(pending_destination)].push_back(*center);
        append_snapshot(std::nullopt);
    }

    if (actions.empty() && prestage_steps == 0) return std::nullopt;

    std::unordered_set<AgentId> omitted;
    for (const IntersectionIntent* intent : participants) {
        const auto& path = paths.at(intent->agent);
        const bool center_visited = std::find(path.begin(), path.end(), intersection.center) != path.end();
        // Releasing an AMR is safe only if its actual next route step is
        // outward. current==exit alone also includes U-turn routes whose next
        // step enters the center; releasing those routes lets them edge-swap
        // with the active schedule.
        const auto arm_index = static_cast<std::size_t>(intent->current);
        if (!center_visited && arm_index < 4
            && follows_natural_egress(intersection, *intent, limit_of(arm_index))
            && (!strict_release_
                || !has_inward_schedule_motion(intersection, intent->current, paths)))
            omitted.insert(intent->agent);
    }
    for (const IntersectionIntent* intent : participants) {
        if (omitted.contains(intent->agent)) continue;
        if (auto trimmed = make_trimmed_egress(intersection, *intent, paths.at(intent->agent)))
            paths[intent->agent] = std::move(*trimmed);
    }

    std::vector<ScheduledPath> scheduled_paths;
    scheduled_paths.reserve(participants.size());

    for (const IntersectionIntent* intent : participants) {
        if (omitted.contains(intent->agent)) continue;
        auto& path = paths[intent->agent];
        // Paths shorter than one move are not inserted into an agent route.
        if (path.size() < 2) continue;
        for (std::size_t i = 0; i + 1 < path.size(); ++i)
            if (!adjacent_or_equal(path[i], path[i + 1], intersection)) return std::nullopt;
        const auto direction = static_cast<std::size_t>(intent->exit);
        if (direction >= 4 || intersection.arms[direction].empty() || limit_of(direction) == 0) return std::nullopt;
        scheduled_paths.push_back({intent->agent, std::move(path),
                                   intersection.arms[direction][limit_of(direction) - 1]});
    }
    if (scheduled_paths.empty()) return std::nullopt;
    return scheduled_paths;
}

}  // namespace lima
