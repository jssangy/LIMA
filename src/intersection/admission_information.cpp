#include "lima/intersection/admission_information.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lima {

void AdmissionInformationAxis::resize(const std::size_t intersections, const std::size_t agents) {
    previous_neighbor_signal_.resize(intersections);
    increasing_steps_.resize(intersections);
    credit_deficit_.resize(intersections);
    grants_.resize(agents, -1);
}

bool AdmissionInformationAxis::lookahead_blocks(
    const IntersectionId source, const IntersectionId downstream,
    const std::span<const std::size_t> stale_loads, const std::span<const int> capacities,
    const std::span<const int> availability) {
    if (config_.lookahead == AdmitLookaheadMode::Off || downstream < 0) return false;
    const auto s = static_cast<std::size_t>(source);
    const auto d = static_cast<std::size_t>(downstream);
    if (s >= capacities.size() || d >= capacities.size()) return false;
    const double source_ratio = static_cast<double>(stale_loads[s])
        / static_cast<double>(std::max(1, capacities[s]));
    const double downstream_ratio = static_cast<double>(stale_loads[d])
        / static_cast<double>(std::max(1, capacities[d]));
    bool blocked = false;
    switch (config_.lookahead) {
    case AdmitLookaheadMode::Off:
        break;
    case AdmitLookaheadMode::Hard:
        blocked = availability[d] <= 0;
        break;
    case AdmitLookaheadMode::Threshold:
        blocked = availability[d] < static_cast<int>(std::ceil(config_.lookahead_parameter));
        break;
    case AdmitLookaheadMode::Ratio:
        blocked = downstream_ratio >= config_.lookahead_parameter;
        break;
    case AdmitLookaheadMode::Differential:
        blocked = source_ratio - downstream_ratio < config_.lookahead_parameter;
        break;
    }
    if (blocked) events_.push_back({"A1", "blocked", source, downstream, 1, 0, 1, 1});
    return blocked;
}

bool AdmissionInformationAxis::aimd_congested(
    const IntersectionId source, const bool local_congested,
    const IntersectionTopology& topology, const std::span<const std::size_t> stale_loads,
    const std::span<const int> capacities) {
    if (local_congested || config_.aimd_signal == AimdSignalMode::Local) return local_congested;
    const auto source_index = static_cast<std::size_t>(source);
    double maximum = 0.0;
    double sum = 0.0;
    int count = 0;
    for (const IntersectionId neighbor : topology.intersections()[source_index].neighbors) {
        if (neighbor < 0) continue;
        const auto n = static_cast<std::size_t>(neighbor);
        const double ratio = static_cast<double>(stale_loads[n])
            / static_cast<double>(std::max(1, capacities[n]));
        maximum = std::max(maximum, ratio);
        sum += ratio;
        ++count;
    }
    if (count == 0) return false;
    const double mean = sum / static_cast<double>(count);
    bool signaled = false;
    if (config_.aimd_signal == AimdSignalMode::NeighborMax) {
        signaled = maximum >= config_.aimd_signal_parameter;
    } else if (config_.aimd_signal == AimdSignalMode::NeighborMean) {
        signaled = mean >= config_.aimd_signal_parameter;
    } else {
        const double aggregate = mean;
        if (aggregate > previous_neighbor_signal_[source_index] + 1e-12)
            ++increasing_steps_[source_index];
        else
            increasing_steps_[source_index] = 0;
        previous_neighbor_signal_[source_index] = aggregate;
        const auto threshold = static_cast<std::uint32_t>(
            std::max(1.0, std::round(config_.aimd_signal_parameter)));
        signaled = increasing_steps_[source_index] >= threshold;
    }
    if (signaled) events_.push_back({"A2", "decrease", source, -1, 1, 0, 1, 1});
    return signaled;
}

void AdmissionInformationAxis::prepare_credits(
    const std::span<const AdmissionCreditRequest> requests, const std::span<const int> availability,
    const std::span<const int> reserve) {
    std::fill(grants_.begin(), grants_.end(), -1);
    if (config_.credit == AdmitCreditMode::Off) return;

    std::vector<std::array<std::vector<AdmissionCreditRequest>, 4>> by_intersection(
        availability.size());
    for (const AdmissionCreditRequest& request : requests) {
        if (request.intersection < 0 || request.direction >= 4 || request.agent < 0) continue;
        by_intersection[static_cast<std::size_t>(request.intersection)][request.direction].push_back(request);
    }
    for (std::size_t iid = 0; iid < by_intersection.size(); ++iid) {
        auto& arms = by_intersection[iid];
        for (auto& arm : arms) {
            std::sort(arm.begin(), arm.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.age != rhs.age ? lhs.age > rhs.age : lhs.agent < rhs.agent;
            });
        }
        int budget = std::max(0, availability[iid] - reserve[iid]);
        std::array<std::size_t, 4> cursor{};
        std::array<std::uint32_t, 4> advertised{};
        while (budget-- > 0) {
            int selected = -1;
            if (config_.credit == AdmitCreditMode::Equal) {
                std::size_t least_granted = std::numeric_limits<std::size_t>::max();
                for (std::size_t d = 0; d < 4; ++d) {
                    if (cursor[d] >= arms[d].size()) continue;
                    if (cursor[d] < least_granted) {
                        least_granted = cursor[d];
                        selected = static_cast<int>(d);
                    }
                }
            } else if (config_.credit == AdmitCreditMode::Demand) {
                double best = -1.0;
                for (std::size_t d = 0; d < 4; ++d) {
                    if (cursor[d] >= arms[d].size()) continue;
                    const double score = static_cast<double>(arms[d].size())
                        / static_cast<double>(cursor[d] + 1);
                    if (score > best) {
                        best = score;
                        selected = static_cast<int>(d);
                    }
                }
            } else {
                const double quantum = config_.credit_parameter > 0.0
                    ? config_.credit_parameter : 1.0;
                for (std::size_t d = 0; d < 4; ++d)
                    if (cursor[d] < arms[d].size()) credit_deficit_[iid][d] += quantum;
                for (std::size_t d = 0; d < 4; ++d) {
                    if (cursor[d] >= arms[d].size()) continue;
                    if (selected < 0 || credit_deficit_[iid][d]
                        > credit_deficit_[iid][static_cast<std::size_t>(selected)])
                        selected = static_cast<int>(d);
                }
                if (selected >= 0)
                    credit_deficit_[iid][static_cast<std::size_t>(selected)] -= 1.0;
            }
            if (selected < 0) break;
            const auto d = static_cast<std::size_t>(selected);
            const AgentId agent = arms[d][cursor[d]++].agent;
            if (static_cast<std::size_t>(agent) < grants_.size())
                grants_[static_cast<std::size_t>(agent)] = static_cast<IntersectionId>(iid);
            ++advertised[d];
            events_.push_back({"A3", "issued", static_cast<IntersectionId>(iid),
                              static_cast<IntersectionId>(iid), 1, 0, 0, 0});
        }
        for (std::size_t d = 0; d < 4; ++d) if (advertised[d] > 0) {
            events_.push_back({"A3", "advertised", static_cast<IntersectionId>(iid),
                              static_cast<IntersectionId>(iid), advertised[d],
                              static_cast<std::uint32_t>(sizeof(std::uint16_t)), 1, 1});
        }
    }
}

bool AdmissionInformationAxis::credit_allows(const AgentId agent,
                                              const IntersectionId entering) const {
    if (config_.credit == AdmitCreditMode::Off) return true;
    return agent >= 0 && static_cast<std::size_t>(agent) < grants_.size()
        && grants_[static_cast<std::size_t>(agent)] == entering;
}

void AdmissionInformationAxis::note_credit_consumed(const AgentId agent,
                                                     const IntersectionId entering) {
    if (config_.credit == AdmitCreditMode::Off || agent < 0
        || static_cast<std::size_t>(agent) >= grants_.size()
        || grants_[static_cast<std::size_t>(agent)] != entering) return;
    grants_[static_cast<std::size_t>(agent)] = -1;
    events_.push_back({"A3", "consumed", entering, entering, 1, 0, 0, 0});
}

std::vector<AdmissionInformationEvent> AdmissionInformationAxis::take_events() {
    auto events = std::move(events_);
    events_.clear();
    return events;
}

}  // namespace lima
