#include "lima/simulation/trace.hpp"

#include <cstdio>
#include <stdexcept>

namespace lima {
namespace {

void write_id_array(std::ofstream& out, const std::vector<AgentId>& ids) {
    out << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ',';
        out << ids[i];
    }
    out << ']';
}

void write_group_events(std::ofstream& out, const std::vector<StepTracer::GroupEvent>& events) {
    out << '[';
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i > 0) out << ',';
        out << "[" << events[i].intersection << ',';
        write_id_array(out, events[i].agents);
        out << ']';
    }
    out << ']';
}

}  // namespace

namespace {

std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char c : text) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '"') escaped += "\\\"";
        else if (static_cast<unsigned char>(c) < 0x20) {
            char buffer[8];
            std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
            escaped += buffer;
        } else escaped += c;
    }
    return escaped;
}

}  // namespace

StepTracer::StepTracer(const std::filesystem::path& path, const GridMap& map, std::string map_file,
                       const std::uint64_t seed, const std::span<const Agent> agents) {
    out_.open(path, std::ios::trunc);
    if (!out_) throw std::runtime_error("cannot open trace file: " + path.string());
    out_ << "{\"type\":\"header\",\"map\":\"" << json_escape(map_file) << "\",\"width\":" << map.width()
         << ",\"height\":" << map.height() << ",\"seed\":" << seed
         << ",\"agents\":" << agents.size() << ",\"starts\":[";
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) out_ << ',';
        out_ << agents[i].position;
    }
    out_ << "],\"goals\":[";
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) out_ << ',';
        out_ << agents[i].goal;
    }
    out_ << "],\"routes\":[";
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) out_ << ',';
        out_ << '[';
        for (std::size_t k = 0; k < agents[i].route.size(); ++k) {
            if (k > 0) out_ << ',';
            out_ << agents[i].route[k];
        }
        out_ << ']';
    }
    out_ << "]}\n";
}

void StepTracer::add_schedule(const IntersectionId intersection, std::vector<AgentId> agents) {
    schedules_.push_back({intersection, std::move(agents)});
}

void StepTracer::add_discharge(const IntersectionId intersection, std::vector<AgentId> agents) {
    discharges_.push_back({intersection, std::move(agents)});
}

void StepTracer::add_completion(const AgentId agent) { completions_.push_back(agent); }

void StepTracer::flush_step(const std::uint64_t timestep, const std::span<const Agent> agents) {
    out_ << "{\"type\":\"step\",\"t\":" << timestep << ",\"pos\":[";
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) out_ << ',';
        out_ << agents[i].position;
    }
    out_ << "],\"active\":[";
    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i > 0) out_ << ',';
        out_ << (agents[i].active ? 1 : 0);
    }
    out_ << "],\"sched\":";
    write_group_events(out_, schedules_);
    out_ << ",\"disch\":";
    write_group_events(out_, discharges_);
    out_ << ",\"done\":";
    write_id_array(out_, completions_);
    out_ << "}\n";
    schedules_.clear();
    discharges_.clear();
    completions_.clear();
}

void StepTracer::finish() {
    out_.flush();
    if (!out_) throw std::runtime_error("trace write failed (disk full?)");
}

}  // namespace lima
