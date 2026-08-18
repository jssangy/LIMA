#include <getopt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <problem.hpp>

namespace {

constexpr std::uint64_t kMask64 = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kTraceScale = 1ULL << 53;
constexpr std::uint16_t kInfinity = std::numeric_limits<std::uint16_t>::max();

std::uint64_t splitmix64(std::uint64_t value)
{
  value = (value + 0x9E3779B97F4A7C15ULL) & kMask64;
  value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL) & kMask64;
  value = ((value ^ (value >> 27)) * 0x94D049BB133111EBULL) & kMask64;
  return (value ^ (value >> 31)) & kMask64;
}

bool command_delayed(int seed, int agent_id, int timestep, double probability)
{
  if (probability <= 0.0) return false;
  std::uint64_t counter =
      (static_cast<std::uint64_t>(seed) ^ 0xA0761D6478BD642FULL) & kMask64;
  counter ^= (static_cast<std::uint64_t>(agent_id + 1)
              * 0xD2B74407B1CE6E93ULL) & kMask64;
  counter ^= (static_cast<std::uint64_t>(timestep + 1)
              * 0xCA5A826395121157ULL) & kMask64;
  const auto threshold = static_cast<std::uint64_t>(
      probability * static_cast<double>(kTraceScale));
  return (splitmix64(counter) >> 11) < threshold;
}

struct Agent {
  int id;
  Node* v_now;
  Node* v_next;
  Node* goal;
  int elapsed;
  int initial_distance;
  float tie_breaker;
};

using Agents = std::vector<Agent*>;

class DistanceTable {
 public:
  explicit DistanceTable(MAPF_Instance* problem)
      : graph_(problem->getG()), nodes_(graph_->getV()),
        compact_index_(graph_->getNodesSize(), -1)
  {
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
      compact_index_.at(nodes_[index]->id) = static_cast<int>(index);
    }
    const std::size_t entries =
        static_cast<std::size_t>(problem->getNum()) * nodes_.size();
    distances_.assign(entries, kInfinity);
    for (int agent = 0; agent < problem->getNum(); ++agent) {
      compute(agent, problem->getGoal(agent));
    }
  }

  std::uint16_t distance(int agent_id, Node* node) const
  {
    const int compact = compact_index_.at(node->id);
    if (compact < 0) return kInfinity;
    return distances_.at(
        static_cast<std::size_t>(agent_id) * nodes_.size()
        + static_cast<std::size_t>(compact));
  }

 private:
  void compute(int agent_id, Node* goal)
  {
    const std::size_t offset =
        static_cast<std::size_t>(agent_id) * nodes_.size();
    std::queue<Node*> open;
    distances_[offset + compact_index_.at(goal->id)] = 0;
    open.push(goal);
    while (!open.empty()) {
      Node* current = open.front();
      open.pop();
      const auto current_distance =
          distances_[offset + compact_index_.at(current->id)];
      if (current_distance >= kInfinity - 1) continue;
      for (Node* next : current->neighbor) {
        const std::size_t index =
            offset + static_cast<std::size_t>(compact_index_.at(next->id));
        if (distances_[index] <= current_distance + 1) continue;
        distances_[index] = static_cast<std::uint16_t>(current_distance + 1);
        open.push(next);
      }
    }
  }

  Graph* graph_;
  Nodes nodes_;
  std::vector<int> compact_index_;
  std::vector<std::uint16_t> distances_;
};

class NativeStochasticPIBT {
 public:
  NativeStochasticPIBT(MAPF_Instance* problem, int max_steps,
                       double delay_probability, int delay_seed,
                       bool exclusive_boundary_goals)
      : problem_(problem), graph_(problem->getG()), random_(problem->getMT()),
        distances_(problem), max_steps_(max_steps),
        delay_probability_(delay_probability), delay_seed_(delay_seed),
        exclusive_boundary_goals_(exclusive_boundary_goals),
        original_agents_(problem->getNum()),
        occupied_now_(graph_->getNodesSize(), nullptr),
        occupied_next_(graph_->getNodesSize(), nullptr),
        completion_steps_(problem->getNum(), 0)
  {
    std::uniform_real_distribution<float> tie(0.0F, 1.0F);
    agents_.reserve(problem_->getNum());
    for (int id = 0; id < problem_->getNum(); ++id) {
      Node* start = problem_->getStart(id);
      Node* goal = problem_->getGoal(id);
      if (occupied_now_.at(start->id) != nullptr) {
        throw std::runtime_error("duplicate start location");
      }
      const auto initial_distance = distances_.distance(id, start);
      if (initial_distance == kInfinity) {
        throw std::runtime_error("unreachable start-goal pair");
      }
      auto* agent = new Agent{
          id, start, nullptr, goal, 0,
          static_cast<int>(initial_distance), tie(*random_)};
      occupied_now_[start->id] = agent;
      agents_.push_back(agent);
    }
  }

  ~NativeStochasticPIBT()
  {
    for (Agent* agent : agents_) delete agent;
  }

  void run()
  {
    auto priority = [](const Agent* left, const Agent* right) {
      if (left->elapsed != right->elapsed) return left->elapsed > right->elapsed;
      if (left->initial_distance != right->initial_distance) {
        return left->initial_distance > right->initial_distance;
      }
      return left->tie_breaker > right->tie_breaker;
    };

    for (int timestep = 0; timestep < max_steps_ && !agents_.empty(); ++timestep) {
      comm_active_agent_steps_ += static_cast<std::uint64_t>(agents_.size());
      std::sort(agents_.begin(), agents_.end(), priority);
      for (Agent* agent : agents_) {
        if (agent->v_next == nullptr) {
          ++comm_root_invocations_;
          plan(agent, nullptr, 0);
        }
      }
      validate_planned_step();

      std::vector<Node*> proposed;
      proposed.reserve(agents_.size());
      std::vector<bool> delayed(agents_.size(), false);
      for (std::size_t index = 0; index < agents_.size(); ++index) {
        Agent* agent = agents_[index];
        proposed.push_back(agent->v_next);
        if (agent->v_next != agent->v_now) {
          ++delay_draws_;
          delayed[index] = command_delayed(
              delay_seed_, agent->id, timestep, delay_probability_);
          delayed_moves_ += static_cast<std::uint64_t>(delayed[index]);
        }
      }

      auto [actual, cancelled] = safe_execute(proposed, delayed);
      safe_executor_interventions_ += cancelled;
      bool diverged = false;
      for (std::size_t index = 0; index < actual.size(); ++index) {
        diverged |= actual[index] != proposed[index];
      }
      deviation_steps_ += static_cast<std::uint64_t>(diverged);
      steps_ = timestep + 1;

      std::fill(occupied_now_.begin(), occupied_now_.end(), nullptr);
      std::fill(occupied_next_.begin(), occupied_next_.end(), nullptr);
      Agents remaining;
      remaining.reserve(agents_.size());
      for (std::size_t index = 0; index < agents_.size(); ++index) {
        Agent* agent = agents_[index];
        agent->v_now = actual[index];
        agent->v_next = nullptr;
        if (exclusive_boundary_goals_ && agent->v_now->boundary
            && agent->v_now != agent->goal) {
          throw std::runtime_error(
              "active agent entered a non-assigned boundary goal");
        }
        state_hash_ ^= static_cast<std::uint64_t>(agent->id + 1);
        state_hash_ *= 1099511628211ULL;
        state_hash_ ^= static_cast<std::uint64_t>(agent->v_now->id + 1);
        state_hash_ *= 1099511628211ULL;
        if (agent->v_now == agent->goal) {
          completion_steps_[agent->id] = steps_;
          ++completed_;
          delete agent;
          continue;
        }
        ++agent->elapsed;
        if (occupied_now_[agent->v_now->id] != nullptr) {
          throw std::runtime_error("vertex conflict after stochastic execution");
        }
        occupied_now_[agent->v_now->id] = agent;
        remaining.push_back(agent);
      }
      agents_ = std::move(remaining);
    }
  }

  int steps() const { return steps_; }
  int completed() const { return completed_; }
  int original_agents() const { return original_agents_; }
  bool solved() const { return completed_ == original_agents_; }
  std::uint64_t delayed_moves() const { return delayed_moves_; }
  std::uint64_t delay_draws() const { return delay_draws_; }
  std::uint64_t deviation_steps() const { return deviation_steps_; }
  std::uint64_t safe_executor_interventions() const
  {
    return safe_executor_interventions_;
  }
  std::uint64_t state_hash() const { return state_hash_; }
  const std::vector<int>& completion_steps() const { return completion_steps_; }
  std::uint64_t comm_active_agent_steps() const { return comm_active_agent_steps_; }
  std::uint64_t comm_root_invocations() const { return comm_root_invocations_; }
  std::uint64_t comm_inheritance_requests() const
  {
    return comm_inheritance_requests_;
  }
  std::uint64_t comm_backtracking_responses() const
  {
    return comm_backtracking_responses_;
  }
  std::uint64_t comm_backtracking_valid() const
  {
    return comm_backtracking_valid_;
  }
  std::uint64_t comm_backtracking_invalid() const
  {
    return comm_backtracking_invalid_;
  }
  int comm_max_propagation_depth() const { return comm_max_propagation_depth_; }

 private:
  bool plan(Agent* agent, Agent* parent, int propagation_depth)
  {
    if (parent != nullptr) {
      ++comm_inheritance_requests_;
      comm_max_propagation_depth_ =
          std::max(comm_max_propagation_depth_, propagation_depth);
    }
    auto candidates = agent->v_now->neighbor;
    candidates.push_back(agent->v_now);
    std::shuffle(candidates.begin(), candidates.end(), *random_);
    std::sort(candidates.begin(), candidates.end(), [&](Node* left, Node* right) {
      const auto left_distance = distances_.distance(agent->id, left);
      const auto right_distance = distances_.distance(agent->id, right);
      if (left_distance != right_distance) return left_distance < right_distance;
      const bool left_occupied = occupied_now_[left->id] != nullptr;
      const bool right_occupied = occupied_now_[right->id] != nullptr;
      if (left_occupied != right_occupied) return !left_occupied;
      return false;
    });

    for (Node* candidate : candidates) {
      if (exclusive_boundary_goals_ && candidate->boundary
          && candidate != agent->goal) {
        continue;
      }
      if (occupied_next_[candidate->id] != nullptr) continue;
      if (parent != nullptr && candidate == parent->v_now) continue;
      occupied_next_[candidate->id] = agent;
      agent->v_next = candidate;
      Agent* inherited = occupied_now_[candidate->id];
      if (inherited != nullptr && inherited->v_next == nullptr) {
        const bool valid = plan(inherited, agent, propagation_depth + 1);
        ++comm_backtracking_responses_;
        if (valid) {
          ++comm_backtracking_valid_;
        } else {
          ++comm_backtracking_invalid_;
          continue;
        }
      }
      return true;
    }
    occupied_next_[agent->v_now->id] = agent;
    agent->v_next = agent->v_now;
    return false;
  }

  void validate_planned_step() const
  {
    std::vector<int> vertex(graph_->getNodesSize(), -1);
    std::unordered_set<std::uint64_t> edges;
    for (Agent* agent : agents_) {
      if (agent->v_next == nullptr) throw std::runtime_error("unplanned agent");
      const int next = agent->v_next->id;
      if (vertex[next] >= 0) throw std::runtime_error("planned vertex conflict");
      vertex[next] = agent->id;
      const auto from = static_cast<std::uint32_t>(agent->v_now->id);
      const auto to = static_cast<std::uint32_t>(next);
      const std::uint64_t reverse =
          (static_cast<std::uint64_t>(to) << 32) | from;
      if (from != to && edges.find(reverse) != edges.end()) {
        throw std::runtime_error("planned edge conflict");
      }
      edges.insert((static_cast<std::uint64_t>(from) << 32) | to);
    }
  }

  std::pair<std::vector<Node*>, std::uint64_t> safe_execute(
      const std::vector<Node*>& proposed, const std::vector<bool>& delayed) const
  {
    std::vector<bool> moving(agents_.size(), false);
    std::vector<int> occupancy(graph_->getNodesSize(), -1);
    for (std::size_t index = 0; index < agents_.size(); ++index) {
      moving[index] = proposed[index] != agents_[index]->v_now && !delayed[index];
      occupancy[agents_[index]->v_now->id] = static_cast<int>(index);
    }
    std::uint64_t cancelled = 0;
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t index = 0; index < agents_.size(); ++index) {
        if (!moving[index]) continue;
        const int occupant = occupancy[proposed[index]->id];
        if (occupant >= 0 && occupant != static_cast<int>(index)
            && !moving[occupant]) {
          moving[index] = false;
          ++cancelled;
          changed = true;
        }
      }

      std::vector<int> target_owner(graph_->getNodesSize(), -1);
      for (std::size_t index = 0; index < agents_.size(); ++index) {
        Node* target = moving[index] ? proposed[index] : agents_[index]->v_now;
        const int previous = target_owner[target->id];
        if (previous < 0) {
          target_owner[target->id] = static_cast<int>(index);
          continue;
        }
        if (moving[index]) {
          moving[index] = false;
          ++cancelled;
          changed = true;
        }
        if (moving[previous]) {
          moving[previous] = false;
          ++cancelled;
          changed = true;
        }
      }

      std::unordered_map<std::uint64_t, std::size_t> transitions;
      for (std::size_t index = 0; index < agents_.size(); ++index) {
        if (!moving[index]) continue;
        const auto from = static_cast<std::uint32_t>(agents_[index]->v_now->id);
        const auto to = static_cast<std::uint32_t>(proposed[index]->id);
        const std::uint64_t reverse =
            (static_cast<std::uint64_t>(to) << 32) | from;
        const auto found = transitions.find(reverse);
        if (found != transitions.end()) {
          moving[index] = false;
          moving[found->second] = false;
          cancelled += 2;
          changed = true;
          break;
        }
        transitions[(static_cast<std::uint64_t>(from) << 32) | to] = index;
      }
    }

    std::vector<Node*> actual;
    actual.reserve(agents_.size());
    std::vector<int> vertex(graph_->getNodesSize(), -1);
    std::unordered_set<std::uint64_t> edges;
    for (std::size_t index = 0; index < agents_.size(); ++index) {
      Node* next = moving[index] ? proposed[index] : agents_[index]->v_now;
      if (vertex[next->id] >= 0) {
        throw std::runtime_error("safe executor produced a vertex conflict");
      }
      vertex[next->id] = static_cast<int>(index);
      const auto from = static_cast<std::uint32_t>(agents_[index]->v_now->id);
      const auto to = static_cast<std::uint32_t>(next->id);
      const std::uint64_t reverse =
          (static_cast<std::uint64_t>(to) << 32) | from;
      if (from != to && edges.find(reverse) != edges.end()) {
        throw std::runtime_error("safe executor produced an edge conflict");
      }
      edges.insert((static_cast<std::uint64_t>(from) << 32) | to);
      actual.push_back(next);
    }
    return {std::move(actual), cancelled};
  }

  MAPF_Instance* problem_;
  Graph* graph_;
  std::mt19937* random_;
  DistanceTable distances_;
  int max_steps_;
  double delay_probability_;
  int delay_seed_;
  bool exclusive_boundary_goals_;
  int original_agents_;
  Agents agents_;
  Agents occupied_now_;
  Agents occupied_next_;
  std::vector<int> completion_steps_;
  int steps_ = 0;
  int completed_ = 0;
  std::uint64_t delayed_moves_ = 0;
  std::uint64_t delay_draws_ = 0;
  std::uint64_t deviation_steps_ = 0;
  std::uint64_t safe_executor_interventions_ = 0;
  std::uint64_t state_hash_ = 1469598103934665603ULL;
  std::uint64_t comm_active_agent_steps_ = 0;
  std::uint64_t comm_root_invocations_ = 0;
  std::uint64_t comm_inheritance_requests_ = 0;
  std::uint64_t comm_backtracking_responses_ = 0;
  std::uint64_t comm_backtracking_valid_ = 0;
  std::uint64_t comm_backtracking_invalid_ = 0;
  int comm_max_propagation_depth_ = 0;
};

double percentile(std::vector<int> values, double quantile)
{
  values.erase(std::remove(values.begin(), values.end(), 0), values.end());
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return static_cast<double>(values[lower]) * (1.0 - fraction)
      + static_cast<double>(values[upper]) * fraction;
}

void help()
{
  std::cout << "Usage: pibt_native_stochastic --instance FILE "
               "--max-steps N --delay-prob P --delay-seed S "
               "[--exclusive-boundary-goals]\n";
}

}  // namespace

int main(int argc, char* argv[])
{
  std::string instance_file;
  int max_steps = 30000;
  double delay_probability = 0.0;
  int delay_seed = 0;
  bool exclusive_boundary_goals = false;
  const option options[] = {
      {"instance", required_argument, nullptr, 'i'},
      {"max-steps", required_argument, nullptr, 'H'},
      {"delay-prob", required_argument, nullptr, 'p'},
      {"delay-seed", required_argument, nullptr, 's'},
      {"exclusive-boundary-goals", no_argument, nullptr, 'b'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };
  int index = 0;
  while (true) {
    const int choice = getopt_long(argc, argv, "i:H:p:s:bh", options, &index);
    if (choice == -1) break;
    if (choice == 'i') instance_file = optarg;
    else if (choice == 'H') max_steps = std::stoi(optarg);
    else if (choice == 'p') delay_probability = std::stod(optarg);
    else if (choice == 's') delay_seed = std::stoi(optarg);
    else if (choice == 'b') exclusive_boundary_goals = true;
    else {
      help();
      return choice == 'h' ? 0 : 2;
    }
  }
  if (instance_file.empty() || max_steps < 1
      || delay_probability < 0.0 || delay_probability > 1.0) {
    help();
    return 2;
  }

  try {
    MAPF_Instance problem(instance_file);
    NativeStochasticPIBT solver(
        &problem, max_steps, delay_probability, delay_seed,
        exclusive_boundary_goals);
    solver.run();

    const auto state_priority_announcements = solver.comm_active_agent_steps();
    const auto decision_announcements = solver.comm_active_agent_steps();
    const auto propagation_events = solver.comm_inheritance_requests()
        + solver.comm_backtracking_responses();
    const auto distributed_logical_events = state_priority_announcements
        + decision_announcements + propagation_events;
    const auto& completions = solver.completion_steps();
    std::uint64_t sum_completion = 0;
    for (int value : completions) sum_completion += static_cast<std::uint64_t>(value);
    const double mean_completion = solver.completed() > 0
        ? static_cast<double>(sum_completion) / solver.completed() : 0.0;

    std::cout << std::setprecision(12)
              << "status=" << (solver.solved() ? "completed" : "step_limit") << '\n'
              << "solved=" << (solver.solved() ? 1 : 0) << '\n'
              << "completed=" << solver.completed() << '/' << solver.original_agents() << '\n'
              << "steps=" << solver.steps() << '\n'
              << "makespan=" << (solver.solved() ? solver.steps() : max_steps) << '\n'
              << "soc=" << sum_completion << '\n'
              << "completion_step_mean=" << mean_completion << '\n'
              << "completion_step_p50=" << percentile(completions, 0.50) << '\n'
              << "completion_step_p90=" << percentile(completions, 0.90) << '\n'
              << "completion_step_p99=" << percentile(completions, 0.99) << '\n'
              << "planning_calls=" << solver.steps() << '\n'
              << "replans=0\n"
              << "native_one_step=1\n"
              << "delayed_moves=" << solver.delayed_moves() << '\n'
              << "delay_draws=" << solver.delay_draws() << '\n'
              << "deviation_steps=" << solver.deviation_steps() << '\n'
              << "safe_executor_interventions="
              << solver.safe_executor_interventions() << '\n'
              << "communication_events=" << distributed_logical_events << '\n'
              << "communication_scope=neighbor-to-neighbor\n"
              << "communication_link_distance_hops=1\n"
              << "pibt_comm_active_agent_steps="
              << solver.comm_active_agent_steps() << '\n'
              << "pibt_comm_root_invocations="
              << solver.comm_root_invocations() << '\n'
              << "pibt_comm_inheritance_requests="
              << solver.comm_inheritance_requests() << '\n'
              << "pibt_comm_backtracking_responses="
              << solver.comm_backtracking_responses() << '\n'
              << "pibt_comm_backtracking_valid="
              << solver.comm_backtracking_valid() << '\n'
              << "pibt_comm_backtracking_invalid="
              << solver.comm_backtracking_invalid() << '\n'
              << "pibt_comm_max_propagation_depth="
              << solver.comm_max_propagation_depth() << '\n'
              << "pibt_comm_state_priority_announcements="
              << state_priority_announcements << '\n'
              << "pibt_comm_decision_announcements="
              << decision_announcements << '\n'
              << "pibt_comm_propagation_events=" << propagation_events << '\n'
              << "pibt_comm_distributed_logical_events="
              << distributed_logical_events << '\n'
              << "vertex_conflicts=0\nedge_conflicts=0\ninvalid_moves=0\n"
              << "boundary_entry_violations=0\n"
              << "trace_spec=splitmix64-agent-step-v1\n"
              << "decision_hash=" << std::hex << solver.state_hash() << std::dec << '\n';
  } catch (const std::exception& error) {
    std::cerr << "pibt_native_stochastic error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
