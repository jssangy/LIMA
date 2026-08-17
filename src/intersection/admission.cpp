#include "lima/intersection/admission.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lima {

void AcknowledgedAimdAdmission::reset(
    const std::size_t agent_count, const std::span<const int> capacities,
    const AcknowledgedAimdConfig config) {
    if (!(config.multiplicative_decrease > 0.0
          && config.multiplicative_decrease <= 1.0))
        throw std::invalid_argument("AIMD multiplicative decrease must be in (0, 1]");
    if (config.additive_increase <= 0.0)
        throw std::invalid_argument("AIMD additive increase must be positive");

    beta_ = config.multiplicative_decrease;
    alpha_ = config.additive_increase;
    capacities_.assign(capacities.begin(), capacities.end());
    windows_.resize(capacities_.size());
    for (std::size_t i = 0; i < capacities_.size(); ++i)
        windows_[i] = static_cast<double>(std::max(1, capacities_[i]));
    grant_target_.assign(agent_count, -1);
    outstanding_.assign(capacities_.size(), 0);
    acknowledged_.assign(capacities_.size(), 0);
    requests_by_intersection_.resize(capacities_.size());
}

void AcknowledgedAimdAdmission::update(
    const std::span<const AimdIntersectionObservation> intersections,
    const std::span<const AimdAdmissionRequest> requests) {
    if (intersections.size() != capacities_.size())
        throw std::invalid_argument("AIMD intersection observation size mismatch");

    std::fill(outstanding_.begin(), outstanding_.end(), 0);
    for (auto& bucket : requests_by_intersection_) bucket.clear();

    std::vector<IntersectionId> requested_target(grant_target_.size(), -1);
    for (const AimdAdmissionRequest& request : requests) {
        if (request.agent < 0 || static_cast<std::size_t>(request.agent) >= grant_target_.size()
            || request.intersection < 0
            || static_cast<std::size_t>(request.intersection) >= intersections.size()) {
            throw std::invalid_argument("AIMD request index out of range");
        }
        requested_target[static_cast<std::size_t>(request.agent)] = request.intersection;
        requests_by_intersection_[static_cast<std::size_t>(request.intersection)].push_back(request);
    }

    // A grant persists only while the same robot requests the same crossing.
    for (std::size_t agent = 0; agent < grant_target_.size(); ++agent) {
        IntersectionId& granted = grant_target_[agent];
        if (granted < 0) continue;
        if (requested_target[agent] != granted) {
            granted = -1;
            continue;
        }
        ++outstanding_[static_cast<std::size_t>(granted)];
    }

    for (std::size_t iid = 0; iid < intersections.size(); ++iid) {
        const int capacity = std::max(1, intersections[iid].capacity);
        capacities_[iid] = capacity;
        const bool structural_congestion =
            (intersections[iid].stalled && !intersections[iid].execution_delayed)
            || intersections[iid].occupied >= capacity
            || intersections[iid].information_congested;
        if (structural_congestion) {
            windows_[iid] = std::max(1.0, beta_ * windows_[iid]);
        } else if (acknowledged_[iid] > 0) {
            windows_[iid] = std::min(
                static_cast<double>(capacity),
                windows_[iid] + alpha_ * static_cast<double>(acknowledged_[iid]));
        }

        auto& bucket = requests_by_intersection_[iid];
        std::sort(bucket.begin(), bucket.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.wait_steps != rhs.wait_steps) return lhs.wait_steps > rhs.wait_steps;
            return lhs.agent < rhs.agent;
        });
        const std::size_t window = static_cast<std::size_t>(
            std::max(1.0, std::floor(windows_[iid])));
        for (const AimdAdmissionRequest& request : bucket) {
            if (outstanding_[iid] >= window) break;
            IntersectionId& granted = grant_target_[static_cast<std::size_t>(request.agent)];
            if (granted >= 0) continue;
            granted = static_cast<IntersectionId>(iid);
            ++outstanding_[iid];
        }
    }
    std::fill(acknowledged_.begin(), acknowledged_.end(), 0);
}

bool AcknowledgedAimdAdmission::granted(
    const AgentId agent, const IntersectionId intersection) const noexcept {
    return agent >= 0 && static_cast<std::size_t>(agent) < grant_target_.size()
        && grant_target_[static_cast<std::size_t>(agent)] == intersection;
}

void AcknowledgedAimdAdmission::acknowledge_entry(
    const AgentId agent, const IntersectionId intersection) noexcept {
    if (!granted(agent, intersection)) return;
    if (intersection >= 0 && static_cast<std::size_t>(intersection) < acknowledged_.size())
        ++acknowledged_[static_cast<std::size_t>(intersection)];
    grant_target_[static_cast<std::size_t>(agent)] = -1;
}

}  // namespace lima
