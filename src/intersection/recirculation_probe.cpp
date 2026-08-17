#include "lima/intersection/recirculation_probe.hpp"

#include <algorithm>

namespace lima {

void RecirculationProbeDetector::resize(const std::size_t intersections) {
    stall_age_.resize(intersections);
    detected_cycles_.resize(intersections);
}

void RecirculationProbeDetector::step(
    const IntersectionTopology& topology, const std::vector<bool>& stalled,
    const std::span<const IntersectionId> wait_for) {
    if (config_.mode == RecirculationProbeMode::Off) return;
    for (std::size_t iid = 0; iid < stalled.size(); ++iid) {
        if (stalled[iid]) ++stall_age_[iid];
        else {
            stall_age_[iid] = 0;
            detected_cycles_[iid].clear();
        }
    }

    std::vector<Probe> next;
    next.reserve(probes_.size() + stalled.size());
    for (std::size_t iid = 0; iid < stalled.size(); ++iid) {
        const auto origin = static_cast<IntersectionId>(iid);
        if (!stalled[iid] || stall_age_[iid] < config_.activation_age
            || wait_for[iid] < 0 || active_origins_.contains(origin)
            || !detected_cycles_[iid].empty()) continue;
        active_origins_.insert(origin);
        next.push_back({origin, origin, config_.ttl, {origin}});
    }
    next.insert(next.end(), std::make_move_iterator(probes_.begin()),
                std::make_move_iterator(probes_.end()));
    probes_.clear();

    const auto adjacent = [&](const IntersectionId from, const IntersectionId to) {
        const auto& neighbors = topology.intersections()[static_cast<std::size_t>(from)].neighbors;
        return std::find(neighbors.begin(), neighbors.end(), to) != neighbors.end();
    };
    for (Probe& probe : next) {
        if (probe.remaining == 0 || probe.current < 0
            || static_cast<std::size_t>(probe.current) >= wait_for.size()) {
            active_origins_.erase(probe.origin);
            continue;
        }
        const IntersectionId target = wait_for[static_cast<std::size_t>(probe.current)];
        if (target < 0 || !adjacent(probe.current, target)) {
            active_origins_.erase(probe.origin);
            continue;
        }
        events_.push_back({"R1", "probe_send", probe.current, target, 1,
                          static_cast<std::uint32_t>(3 * sizeof(std::uint16_t)), 1, 1});
        --probe.remaining;
        if (target == probe.origin && probe.path.size() >= 2) {
            detected_cycles_[static_cast<std::size_t>(probe.origin)] = probe.path;
            events_.push_back({"R1", "cycle_detected", probe.origin, probe.origin, 1,
                              0, static_cast<std::uint16_t>(probe.path.size()),
                              static_cast<std::uint16_t>(probe.path.size())});
            active_origins_.erase(probe.origin);
            continue;
        }
        if (std::find(probe.path.begin(), probe.path.end(), target) != probe.path.end()
            || probe.remaining == 0) {
            active_origins_.erase(probe.origin);
            continue;
        }
        probe.current = target;
        probe.path.push_back(target);
        probes_.push_back(std::move(probe));
    }
}

std::vector<AdmissionInformationEvent> RecirculationProbeDetector::take_events() {
    auto events = std::move(events_);
    events_.clear();
    return events;
}

}  // namespace lima
