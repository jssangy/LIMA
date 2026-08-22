#include "lima/simulation/terminal_egress.hpp"
#include <algorithm>
namespace lima {
void TerminalEgressReservations::resize(
    const std::size_t agent_count, const std::size_t cell_count) {
    leases_.assign(agent_count, {});
    terminal_owner_.assign(cell_count, kNoAgent);
    inward_owner_.assign(cell_count, kNoAgent);
}
void TerminalEgressReservations::clear() noexcept {
    std::fill(leases_.begin(), leases_.end(), Lease{});
    std::fill(terminal_owner_.begin(), terminal_owner_.end(), kNoAgent);
    std::fill(inward_owner_.begin(), inward_owner_.end(), kNoAgent);
}
void TerminalEgressReservations::release(const AgentId agent) noexcept {
    const auto index = static_cast<std::size_t>(agent);
    if (index >= leases_.size()) return;
    const Lease lease = leases_[index];
    if (lease.terminal >= 0
        && static_cast<std::size_t>(lease.terminal) < terminal_owner_.size()
        && terminal_owner_[static_cast<std::size_t>(lease.terminal)] == agent)
        terminal_owner_[static_cast<std::size_t>(lease.terminal)] = kNoAgent;
    if (lease.inward >= 0
        && static_cast<std::size_t>(lease.inward) < inward_owner_.size()
        && inward_owner_[static_cast<std::size_t>(lease.inward)] == agent)
        inward_owner_[static_cast<std::size_t>(lease.inward)] = kNoAgent;
    leases_[index] = {};
}
void TerminalEgressReservations::begin_cycle(
    const std::span<const Agent> agents, const bool enabled) {
    if (!enabled) { clear(); return; }
    const std::size_t count = std::min(agents.size(), leases_.size());
    for (std::size_t index = 0; index < count; ++index) {
        const Lease lease = leases_[index];
        if (lease.terminal == kInvalidCell) continue;
        const Agent& agent = agents[index];
        const bool approaching = agent.active && agent.position == lease.inward
            && agent.goal == lease.terminal && agent.intended_cell() == lease.terminal;
        const bool servicing = agent.active && agent.position == lease.terminal;
        if (!approaching && !servicing) release(static_cast<AgentId>(index));
    }
    for (std::size_t index = count; index < leases_.size(); ++index)
        if (leases_[index].terminal != kInvalidCell) release(static_cast<AgentId>(index));
}
bool TerminalEgressReservations::try_acquire(
    const AgentId agent, const CellId terminal, const CellId inward) {
    const auto index = static_cast<std::size_t>(agent);
    if (agent < 0 || index >= leases_.size() || terminal < 0 || inward < 0
        || static_cast<std::size_t>(terminal) >= terminal_owner_.size()
        || static_cast<std::size_t>(inward) >= inward_owner_.size()) return false;
    const Lease current = leases_[index];
    if (current.terminal != kInvalidCell)
        return current.terminal == terminal && current.inward == inward;
    const AgentId terminal_owner = terminal_owner_[static_cast<std::size_t>(terminal)];
    const AgentId inward_owner = inward_owner_[static_cast<std::size_t>(inward)];
    if ((terminal_owner != kNoAgent && terminal_owner != agent)
        || (inward_owner != kNoAgent && inward_owner != agent)) return false;
    leases_[index] = {terminal, inward};
    terminal_owner_[static_cast<std::size_t>(terminal)] = agent;
    inward_owner_[static_cast<std::size_t>(inward)] = agent;
    return true;
}
bool TerminalEgressReservations::allows(
    const AgentId agent, const CellId destination) const noexcept {
    if (destination < 0 || static_cast<std::size_t>(destination) >= terminal_owner_.size())
        return true;
    const AgentId terminal_owner = terminal_owner_[static_cast<std::size_t>(destination)];
    const AgentId inward_owner = inward_owner_[static_cast<std::size_t>(destination)];
    return (terminal_owner == kNoAgent || terminal_owner == agent)
        && (inward_owner == kNoAgent || inward_owner == agent);
}
CellId TerminalEgressReservations::return_cell(
    const AgentId agent, const CellId position) const noexcept {
    const CellId terminal = terminal_cell(agent);
    return terminal != kInvalidCell && position == terminal ? inward_cell(agent) : kInvalidCell;
}
CellId TerminalEgressReservations::terminal_cell(const AgentId agent) const noexcept {
    const auto index = static_cast<std::size_t>(agent);
    return agent >= 0 && index < leases_.size() ? leases_[index].terminal : kInvalidCell;
}
CellId TerminalEgressReservations::inward_cell(const AgentId agent) const noexcept {
    const auto index = static_cast<std::size_t>(agent);
    return agent >= 0 && index < leases_.size() ? leases_[index].inward : kInvalidCell;
}
}  // namespace lima
