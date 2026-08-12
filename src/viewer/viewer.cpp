#include "lima/viewer/viewer.hpp"

#include <SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lima {
namespace {

class SdlContext {
public:
    SdlContext() {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) throw std::runtime_error(SDL_GetError());
    }
    ~SdlContext() { SDL_Quit(); }
};

class Window {
public:
    Window(const int width, const int height) {
        window_ = SDL_CreateWindow("LIMA C++", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height,
                                   SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if (window_ == nullptr) throw std::runtime_error(SDL_GetError());
        SDL_SetWindowMinimumSize(window_, 800, 600);
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (renderer_ == nullptr) renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
        if (renderer_ == nullptr) throw std::runtime_error(SDL_GetError());
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    }
    ~Window() {
        if (renderer_ != nullptr) SDL_DestroyRenderer(renderer_);
        if (window_ != nullptr) SDL_DestroyWindow(window_);
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] SDL_Window* window() const noexcept { return window_; }
    [[nodiscard]] SDL_Renderer* renderer() const noexcept { return renderer_; }

private:
    SDL_Window* window_{};
    SDL_Renderer* renderer_{};
};

void color(SDL_Renderer* renderer, const std::array<std::uint8_t, 4> rgba) {
    SDL_SetRenderDrawColor(renderer, rgba[0], rgba[1], rgba[2], rgba[3]);
}

std::array<std::uint8_t, 4> agent_color(const AgentId id) {
    constexpr double phi_conjugate = 0.6180339887498948;
    constexpr double saturation = 0.85;
    constexpr double value = 0.95;
    const double hue = std::fmod(static_cast<double>(id) * phi_conjugate, 1.0);
    const double scaled = hue * 6.0;
    const int sector = static_cast<int>(std::floor(scaled));
    const double fraction = scaled - static_cast<double>(sector);
    const double p = value * (1.0 - saturation);
    const double q = value * (1.0 - saturation * fraction);
    const double t = value * (1.0 - saturation * (1.0 - fraction));
    double r = value;
    double g = t;
    double b = p;
    switch (sector % 6) {
        case 0: r = value; g = t; b = p; break;
        case 1: r = q; g = value; b = p; break;
        case 2: r = p; g = value; b = t; break;
        case 3: r = p; g = q; b = value; break;
        case 4: r = t; g = p; b = value; break;
        case 5: r = value; g = p; b = q; break;
    }
    return {static_cast<std::uint8_t>(r * 255.0), static_cast<std::uint8_t>(g * 255.0),
            static_cast<std::uint8_t>(b * 255.0), 255};
}

void fill_circle(SDL_Renderer* renderer, const int center_x, const int center_y, const int radius) {
    for (int y = -radius; y <= radius; ++y) {
        const int half_width = static_cast<int>(std::sqrt(static_cast<double>(radius * radius - y * y)));
        SDL_RenderDrawLine(renderer, center_x - half_width, center_y + y, center_x + half_width, center_y + y);
    }
}

}  // namespace

int run_viewer_impl(Simulator* simulator, const GridMap& map, const SolutionTrace* replay,
                    SolutionTrace* recorder, ViewerOptions options) {
    SdlContext sdl;
    constexpr int window_width = 1600;
    constexpr int window_height = 900;
    Window window(window_width, window_height);
    SDL_Renderer* renderer = window.renderer();

    double zoom = 8.0;
    double view_offset_x = 0.0;
    double view_offset_y = 0.0;
    const auto fit_map = [&]() {
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(window.window(), &width, &height);
        constexpr double padding = 32.0;
        zoom = std::clamp(std::min((static_cast<double>(width) - padding * 2.0) / map.width(),
                                   (static_cast<double>(height) - padding * 2.0) / map.height()),
                          0.5, 50.0);
        view_offset_x = (static_cast<double>(map.width()) * zoom - width) / 2.0;
        view_offset_y = (static_cast<double>(map.height()) * zoom - height) / 2.0;
    };
    const auto zoom_at = [&](const double factor, const int mouse_x, const int mouse_y) {
        const double old_zoom = zoom;
        zoom = std::clamp(zoom * factor, 0.5, 50.0);
        const double world_x = (mouse_x + view_offset_x) / old_zoom;
        const double world_y = (mouse_y + view_offset_y) / old_zoom;
        view_offset_x = world_x * zoom - mouse_x;
        view_offset_y = world_y * zoom - mouse_y;
    };
    fit_map();

    bool running = true;
    bool paused = false;
    bool single_step = false;
    bool panning = false;
    bool show_goal_lines = false;
    std::size_t replay_cursor = 0;
    auto previous = std::chrono::steady_clock::now();
    double accumulator = 0.0;

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_MOUSEWHEEL) {
                int mouse_x = 0;
                int mouse_y = 0;
                SDL_GetMouseState(&mouse_x, &mouse_y);
                zoom_at(event.wheel.y > 0 ? 1.2 : 1.0 / 1.2, mouse_x, mouse_y);
            } else if (event.type == SDL_MOUSEBUTTONDOWN
                       && (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_MIDDLE)) {
                int width = 0;
                int height = 0;
                SDL_GetWindowSize(window.window(), &width, &height);
                (void)height;
                const SDL_Rect goal_line_button{width - 60, 16, 44, 32};
                const SDL_Point click{event.button.x, event.button.y};
                if (event.button.button == SDL_BUTTON_LEFT
                    && SDL_PointInRect(&click, &goal_line_button) == SDL_TRUE) {
                    show_goal_lines = !show_goal_lines;
                    panning = false;
                } else {
                    panning = true;
                }
            } else if (event.type == SDL_MOUSEBUTTONUP
                       && (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_MIDDLE)) {
                panning = false;
            } else if (event.type == SDL_MOUSEMOTION && panning) {
                view_offset_x -= event.motion.xrel;
                view_offset_y -= event.motion.yrel;
            } else if (event.type == SDL_KEYDOWN) {
                int mouse_x = 0;
                int mouse_y = 0;
                SDL_GetMouseState(&mouse_x, &mouse_y);
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE: running = false; break;
                    case SDLK_SPACE:
                        paused = !paused;
                        accumulator = 0.0;
                        previous = std::chrono::steady_clock::now();
                        break;
                    case SDLK_RIGHT:
                        paused = true;
                        if (replay != nullptr) replay_cursor = std::min(replay_cursor + 1, replay->frame_count() - 1);
                        else single_step = true;
                        accumulator = 0.0;
                        previous = std::chrono::steady_clock::now();
                        break;
                    case SDLK_LEFT:
                        if (replay != nullptr) {
                            paused = true;
                            if (replay_cursor > 0) --replay_cursor;
                            accumulator = 0.0;
                            previous = std::chrono::steady_clock::now();
                        }
                        break;
                    case SDLK_HOME:
                        if (replay != nullptr) {
                            paused = true;
                            replay_cursor = 0;
                            accumulator = 0.0;
                            previous = std::chrono::steady_clock::now();
                        }
                        break;
                    case SDLK_END:
                        if (replay != nullptr) {
                            paused = true;
                            replay_cursor = replay->frame_count() - 1;
                            accumulator = 0.0;
                            previous = std::chrono::steady_clock::now();
                        }
                        break;
                    case SDLK_UP:
                        options.steps_per_second = options.steps_per_second == 0.0
                            ? 20.0 : std::min(240.0, options.steps_per_second * 1.5);
                        break;
                    case SDLK_DOWN:
                        options.steps_per_second = options.steps_per_second == 0.0
                            ? 240.0 : std::max(0.5, options.steps_per_second / 1.5);
                        break;
                    case SDLK_EQUALS:
                    case SDLK_KP_PLUS: zoom_at(1.2, mouse_x, mouse_y); break;
                    case SDLK_MINUS:
                    case SDLK_KP_MINUS: zoom_at(1.0 / 1.2, mouse_x, mouse_y); break;
                    case SDLK_f: fit_map(); break;
                    case SDLK_g: show_goal_lines = !show_goal_lines; break;
                    default: break;
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        if (!paused) accumulator += std::chrono::duration<double>(now - previous).count();
        else accumulator = 0.0;
        previous = now;
        const bool unlimited = options.steps_per_second == 0.0;
        const double interval = unlimited ? 0.0 : 1.0 / options.steps_per_second;
        if (replay != nullptr) {
            if (unlimited && !paused && replay_cursor + 1 < replay->frame_count()) {
                ++replay_cursor;
                accumulator = 0.0;
            } else while (!paused && replay_cursor + 1 < replay->frame_count() && accumulator >= interval) {
                ++replay_cursor;
                accumulator -= interval;
            }
            if (replay_cursor + 1 >= replay->frame_count()) paused = true;
        } else {
            if (single_step && !simulator->done() && simulator->stats().timestep < options.max_steps) {
                if (!simulator->step()) paused = true;
                else if (recorder != nullptr) recorder->append(simulator->agents());
                single_step = false;
                accumulator = 0.0;
            }
            if (unlimited && !paused && !single_step && !simulator->done()
                && simulator->stats().timestep < options.max_steps) {
                if (!simulator->step()) paused = true;
                else if (recorder != nullptr) recorder->append(simulator->agents());
                accumulator = 0.0;
            } else while (!paused && !simulator->done() && simulator->stats().timestep < options.max_steps
                          && accumulator >= interval) {
                if (!simulator->step()) {
                    paused = true;
                    break;
                }
                if (recorder != nullptr) recorder->append(simulator->agents());
                accumulator -= interval;
            }
            if (simulator->done() || simulator->stats().timestep >= options.max_steps) paused = true;
        }

        color(renderer, {32, 32, 32, 255});
        SDL_RenderClear(renderer);
        const auto screen_x = [&](const double map_x) { return static_cast<int>(std::lround(map_x * zoom - view_offset_x)); };
        const auto screen_y = [&](const double map_y) { return static_cast<int>(std::lround(map_y * zoom - view_offset_y)); };
        const int cell_size = std::max(1, static_cast<int>(std::ceil(zoom)));
        int viewport_width = 0;
        int viewport_height = 0;
        SDL_GetWindowSize(window.window(), &viewport_width, &viewport_height);
        const int first_x = std::clamp(static_cast<int>(std::floor(view_offset_x / zoom)) - 1, 0, map.width());
        const int first_y = std::clamp(static_cast<int>(std::floor(view_offset_y / zoom)) - 1, 0, map.height());
        const int last_x = std::clamp(static_cast<int>(std::ceil((view_offset_x + viewport_width) / zoom)) + 1, 0, map.width());
        const int last_y = std::clamp(static_cast<int>(std::ceil((view_offset_y + viewport_height) / zoom)) + 1, 0, map.height());
        for (int y = first_y; y < last_y; ++y) for (int x = first_x; x < last_x; ++x) {
            const Coord coordinate{x, y};
            if (map.traversable(coordinate)) continue;
            SDL_Rect rect{screen_x(x) + 1, screen_y(y) + 1, std::max(1, cell_size - 2), std::max(1, cell_size - 2)};
            color(renderer, {160, 160, 160, 255});
            SDL_RenderFillRect(renderer, &rect);
        }
        const auto draw_goal = [&](const AgentId id, const CellId goal_cell) {
            const Coord goal = map.coord(goal_cell);
            SDL_Rect rect{screen_x(goal.x), screen_y(goal.y), cell_size, cell_size};
            color(renderer, agent_color(id));
            SDL_RenderFillRect(renderer, &rect);
        };
        const auto draw_agent_at = [&](const AgentId id, const double map_x, const double map_y, const bool scheduled) {
            const int center_x = screen_x(map_x + 0.5);
            const int center_y = screen_y(map_y + 0.5);
            const int radius = std::max(1, cell_size / 2 - 2);
            color(renderer, agent_color(id));
            fill_circle(renderer, center_x, center_y, radius);
            if (scheduled) {
                color(renderer, {255, 255, 255, 255});
                fill_circle(renderer, center_x, center_y, std::max(1, radius / 2));
            }
        };
        const auto draw_goal_line = [&](const AgentId id, const double map_x, const double map_y,
                                        const CellId goal_cell) {
            const Coord goal = map.coord(goal_cell);
            auto line_color = agent_color(id);
            line_color[3] = 180;
            color(renderer, line_color);
            const int from_x = screen_x(map_x + 0.5);
            const int from_y = screen_y(map_y + 0.5);
            const int to_x = screen_x(goal.x + 0.5);
            const int to_y = screen_y(goal.y + 0.5);
            SDL_RenderDrawLine(renderer, from_x, from_y, to_x, to_y);
            SDL_RenderDrawLine(renderer, from_x + 1, from_y, to_x + 1, to_y);
        };
        if (replay != nullptr) {
            const auto positions = replay->frame(replay_cursor);
            const auto goals = replay->goal_frame(replay_cursor);
            const std::size_t next_cursor = std::min(replay_cursor + 1, replay->frame_count() - 1);
            const auto next_positions = replay->frame(next_cursor);
            const double raw_alpha = (!unlimited && !paused && next_cursor != replay_cursor)
                ? std::clamp(accumulator / interval, 0.0, 1.0) : 0.0;
            if (show_goal_lines) {
                for (std::size_t i = 0; i < replay->agent_count(); ++i) {
                    if (!replay->active(replay_cursor, i)) continue;
                    if (positions[i] == goals[i]) continue;
                    const Coord from = map.coord(positions[i]);
                    const Coord to = map.coord(next_positions[i]);
                    const double x = static_cast<double>(from.x) + (to.x - from.x) * raw_alpha;
                    const double y = static_cast<double>(from.y) + (to.y - from.y) * raw_alpha;
                    draw_goal_line(static_cast<AgentId>(i), x, y, goals[i]);
                }
            }
            for (std::size_t i = 0; i < replay->agent_count(); ++i)
                if (replay->active(replay_cursor, i)) draw_goal(static_cast<AgentId>(i), goals[i]);
            for (std::size_t i = 0; i < replay->agent_count(); ++i) {
                if (!replay->active(replay_cursor, i)) continue;
                const Coord from = map.coord(positions[i]);
                const Coord to = map.coord(next_positions[i]);
                const double x = static_cast<double>(from.x) + (to.x - from.x) * raw_alpha;
                const double y = static_cast<double>(from.y) + (to.y - from.y) * raw_alpha;
                draw_agent_at(static_cast<AgentId>(i), x, y, false);
            }
        } else {
            if (show_goal_lines) {
                for (const Agent& agent : simulator->agents()) if (agent.active) {
                    const Coord position = map.coord(agent.position);
                    draw_goal_line(agent.id, position.x, position.y, agent.goal);
                }
            }
            for (const Agent& agent : simulator->agents())
                if (agent.active) draw_goal(agent.id, agent.goal);
            for (const Agent& agent : simulator->agents()) if (agent.active) {
                const Coord position = map.coord(agent.position);
                draw_agent_at(agent.id, position.x, position.y, agent.scheduled());
            }
        }

        const SDL_Rect goal_line_button{viewport_width - 60, 16, 44, 32};
        color(renderer, show_goal_lines ? std::array<std::uint8_t, 4>{55, 80, 62, 235}
                                        : std::array<std::uint8_t, 4>{55, 55, 55, 220});
        SDL_RenderFillRect(renderer, &goal_line_button);
        color(renderer, show_goal_lines ? std::array<std::uint8_t, 4>{110, 220, 130, 255}
                                        : std::array<std::uint8_t, 4>{155, 155, 155, 255});
        SDL_RenderDrawRect(renderer, &goal_line_button);
        SDL_RenderDrawLine(renderer, goal_line_button.x + 11, goal_line_button.y + 22,
                          goal_line_button.x + 33, goal_line_button.y + 10);
        fill_circle(renderer, goal_line_button.x + 10, goal_line_button.y + 22, 3);
        fill_circle(renderer, goal_line_button.x + 34, goal_line_button.y + 10, 3);
        SDL_RenderPresent(renderer);

        std::uint64_t completed = 0;
        std::size_t total_agents = 0;
        std::size_t timestep = 0;
        bool lifelong = false;
        if (replay != nullptr) {
            const auto positions = replay->frame(replay_cursor);
            total_agents = replay->agent_count();
            timestep = replay_cursor;
            lifelong = replay->lifelong();
            if (lifelong) completed = replay->tasks_completed(replay_cursor);
            else for (std::size_t i = 0; i < total_agents; ++i)
                completed += !replay->active(replay_cursor, i)
                    || positions[i] == replay->goals()[i] ? 1 : 0;
        } else {
            lifelong = simulator->lifelong();
            completed = lifelong ? simulator->stats().completed_tasks : simulator->stats().completed;
            total_agents = simulator->agents().size();
            timestep = simulator->stats().timestep;
        }
        const std::string title = std::string(replay != nullptr ? "LIMA Replay | " : "LIMA Realtime | ")
            + (lifelong ? std::to_string(completed) + " tasks completed"
                        : std::to_string(completed) + "/" + std::to_string(total_agents) + " completed")
            + " | step "
            + std::to_string(timestep) + (replay != nullptr ? "/" + std::to_string(replay->frame_count() - 1) : "")
            + " | " + (paused ? "paused" : (unlimited ? "unlimited" :
                std::to_string(static_cast<int>(options.steps_per_second)) + " step/s"))
            + " | goals " + (show_goal_lines ? "on" : "off")
            + " | zoom " + std::to_string(static_cast<int>(std::lround(zoom * 100.0))) + "%";
        SDL_SetWindowTitle(window.window(), title.c_str());
        if (!unlimited) SDL_Delay(1);
    }
    return 0;
}

int run_viewer(Simulator& simulator, const ViewerOptions options, SolutionTrace* recorder) {
    return run_viewer_impl(&simulator, simulator.map(), nullptr, recorder, options);
}

int run_replay(const GridMap& map, const SolutionTrace& trace, const ViewerOptions options) {
    if (trace.frame_count() == 0) throw std::runtime_error("cannot replay an empty solution");
    return run_viewer_impl(nullptr, map, &trace, nullptr, options);
}

}  // namespace lima
