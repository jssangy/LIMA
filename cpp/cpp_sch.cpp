#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using Move = std::pair<int,int>;

static constexpr int MAX_STACKS = 4;   // 3-way/4-way intersection
static constexpr int MAX_CAP    = 16;  // Can be increased if lane_len exceeds 16
static constexpr int MAX_TYPES  = 16;  // src/dst 0..15, types also allowed up to 0..15
static constexpr uint8_t EMPTY  = 16;  // empty used as zobrist index

// -------------------------
// SplitMix64 (deterministic RNG for zobrist)
// -------------------------
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// zobrist[stack][pos][val]
static std::array<std::array<std::array<uint64_t, MAX_TYPES+1>, MAX_CAP>, MAX_STACKS> Z1;
static std::array<std::array<std::array<uint64_t, MAX_TYPES+1>, MAX_CAP>, MAX_STACKS> Z2;
static bool Z_INIT = false;

static void init_zobrist() {
    if (Z_INIT) return;
    uint64_t seed1 = 0x12345678ABCDEF01ULL;
    uint64_t seed2 = 0x0FEDCBA987654321ULL;

    for (int s=0; s<MAX_STACKS; s++) {
        for (int p=0; p<MAX_CAP; p++) {
            for (int v=0; v<=MAX_TYPES; v++) {
                Z1[s][p][v] = splitmix64(seed1);
                Z2[s][p][v] = splitmix64(seed2);
            }
        }
    }
    Z_INIT = true;
}

struct Key128 {
    uint64_t a;
    uint64_t b;
    bool operator==(const Key128& o) const { return a==o.a && b==o.b; }
};

struct KeyHasher {
    size_t operator()(const Key128& k) const noexcept {
        // 64-bit mix
        uint64_t x = k.a ^ (k.b + 0x9e3779b97f4a7c15ULL + (k.a<<6) + (k.a>>2));
        x ^= (x >> 33);
        x *= 0xff51afd7ed558ccdULL;
        x ^= (x >> 33);
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= (x >> 33);
        return (size_t)x;
    }
};

struct State {
    int n = 0;
    std::array<uint8_t, MAX_STACKS> cap{};
    std::array<uint8_t, MAX_STACKS> sz{};
    // cells[stack][pos] bottom->top, pos < cap[stack]
    std::array<std::array<uint8_t, MAX_CAP>, MAX_STACKS> cells{};
    Key128 h{0,0};

    uint8_t overflow_mask = 0; // bit t is overflow type (counts[t] > cap[t]) -- counts are constant
};

static inline void hash_set(State& st, int s, int p, uint8_t oldv, uint8_t newv) {
    st.h.a ^= Z1[s][p][oldv] ^ Z1[s][p][newv];
    st.h.b ^= Z2[s][p][oldv] ^ Z2[s][p][newv];
}

static inline bool is_overflow_type(const State& st, int t) {
    return (st.overflow_mask & (1u << t)) != 0;
}

static inline bool stack_target_ready(const State& st, int dst) {
    // "Pure" stack: empty or all are of dst type
    for (int i=0; i<st.sz[dst]; i++) {
        if (st.cells[dst][i] != (uint8_t)dst) return false;
    }
    return true;
}

static bool is_sorted(const State& st) {
    // If no overflow, check for strict sorted only
    if (st.overflow_mask == 0) {
        for (int sid=0; sid<st.n; sid++) {
            for (int i=0; i<st.sz[sid]; i++) {
                if (st.cells[sid][i] != (uint8_t)sid) return false;
            }
        }
        return true;
    }

    // overflow rules
    for (int sid=0; sid<st.n; sid++) {
        const bool is_of = is_overflow_type(st, sid);
        if (is_of) {
            if (st.sz[sid] != st.cap[sid]) return false;
            for (int i=0; i<st.sz[sid]; i++) {
                if (st.cells[sid][i] != (uint8_t)sid) return false;
            }
        } else {
            int idx = 0;
            while (idx < st.sz[sid] && st.cells[sid][idx] == (uint8_t)sid) idx++;
            while (idx < st.sz[sid] && is_overflow_type(st, (int)st.cells[sid][idx])) idx++;
            if (idx != st.sz[sid]) return false;
        }
    }
    return true;
}

// Scale Python H2 by 2 (int) to remove float operations
// 1.5 -> 3, 1.0 -> 2
static inline int heuristic_h2_scaled2(const State& st) {
    int h2 = 0;
    for (int sid=0; sid<st.n; sid++) {
        bool clean = true;
        const int cap = st.cap[sid];
        for (int depth=0; depth<st.sz[sid]; depth++) {
            uint8_t ball = st.cells[sid][depth];
            if (clean) {
                if (ball != (uint8_t)sid) {
                    clean = false;
                    h2 += (cap - depth) * 3;
                }
            } else {
                h2 += 2;
            }
        }
    }
    return h2;
}

struct Undo {
    uint8_t ball;
    uint8_t src;
    uint8_t dst;
    uint8_t src_pos; // old top pos
    uint8_t dst_pos; // new top pos
};

static inline void do_move(State& st, int src, int dst, Undo& u) {
    u.src = (uint8_t)src;
    u.dst = (uint8_t)dst;
    u.src_pos = (uint8_t)(st.sz[src] - 1);
    u.dst_pos = (uint8_t)(st.sz[dst]);
    u.ball = st.cells[src][u.src_pos];

    // src_pos: ball -> EMPTY
    hash_set(st, src, u.src_pos, u.ball, EMPTY);
    // dst_pos: EMPTY -> ball
    hash_set(st, dst, u.dst_pos, EMPTY, u.ball);

    st.sz[src]--;
    st.cells[dst][u.dst_pos] = u.ball;
    st.sz[dst]++;
}

static inline void undo_move(State& st, const Undo& u) {
    const int src = (int)u.src;
    const int dst = (int)u.dst;

    // sizes rollback
    st.sz[dst]--;
    st.sz[src]++;

    // dst_pos: ball -> EMPTY
    hash_set(st, dst, (int)u.dst_pos, u.ball, EMPTY);
    // src_pos: EMPTY -> ball
    hash_set(st, src, (int)u.src_pos, EMPTY, u.ball);

    st.cells[src][u.src_pos] = u.ball;
}

// move ordering (aggressive sorting as result identity is not required)
static inline void gen_moves(
    const State& st,
    int last_src,
    int last_dst,
    bool use_deterministic_move,
    std::vector<Move>& out
) {
    out.clear();

    // (1) deterministic: if top can be sent directly to target and target is pure, only that one
    if (use_deterministic_move) {
        for (int src=0; src<st.n; src++) {
            if (st.sz[src] == 0) continue;
            uint8_t ball = st.cells[src][st.sz[src]-1];
            int target = (int)ball;
            if (target < 0 || target >= st.n) continue;
            if (target == src) continue;
            if (st.sz[target] >= st.cap[target]) continue;
            if (!stack_target_ready(st, target)) continue;
            if (!(src == last_dst && target == last_src)) {
                out.push_back({src, target});
                return;
            }
        }
    }

    std::vector<Move> direct, same, empty, rest;
    direct.reserve(8); same.reserve(8); empty.reserve(8); rest.reserve(16);

    for (int src=0; src<st.n; src++) {
        if (st.sz[src] == 0) continue;
        uint8_t ball = st.cells[src][st.sz[src]-1];

        for (int dst=0; dst<st.n; dst++) {
            if (src == dst) continue;
            if (st.sz[dst] >= st.cap[dst]) continue;
            if (src == last_dst && dst == last_src) continue; // prevent ping-pong

            // priority classification
            if (dst == (int)ball && stack_target_ready(st, dst)) {
                direct.push_back({src, dst});
            } else if (st.sz[dst] > 0 && st.cells[dst][st.sz[dst]-1] == ball) {
                same.push_back({src, dst});
            } else if (st.sz[dst] == 0) {
                empty.push_back({src, dst});
            } else {
                rest.push_back({src, dst});
            }
        }
    }

    // direct -> same -> empty -> rest
    out.insert(out.end(), direct.begin(), direct.end());
    out.insert(out.end(), same.begin(), same.end());
    out.insert(out.end(), empty.begin(), empty.end());
    out.insert(out.end(), rest.begin(), rest.end());
}

struct DfsRes {
    int f2;      // scaled by 2
    bool found;
};

static DfsRes dfs_ida(
    State& st,
    int g,                 // depth
    int threshold2,
    std::unordered_map<Key128, int, KeyHasher>& visited,
    std::vector<Move>& path,
    bool use_deterministic_move
) {
    auto it = visited.find(st.h);
    if (it != visited.end() && it->second <= g) {
        return { std::numeric_limits<int>::max(), false };
    }
    visited[st.h] = g;

    const int h2 = heuristic_h2_scaled2(st);
    const int f2 = g*2 + h2;
    if (f2 > threshold2) return { f2, false };

    if (is_sorted(st)) {
        return { f2, true };
    }

    int last_src = -1, last_dst = -1;
    if (!path.empty()) {
        last_src = path.back().first;
        last_dst = path.back().second;
    }

    std::vector<Move> moves;
    gen_moves(st, last_src, last_dst, use_deterministic_move, moves);

    int min_over = std::numeric_limits<int>::max();

    for (const auto& mv : moves) {
        Undo u{};
        do_move(st, mv.first, mv.second, u);
        path.push_back(mv);

        auto res = dfs_ida(st, g+1, threshold2, visited, path, use_deterministic_move);
        if (res.found) return res;
        min_over = std::min(min_over, res.f2);

        path.pop_back();
        undo_move(st, u);
    }

    return { min_over, false };
}

static std::vector<Move> solve_h2_base_cpp(
    const std::vector<std::vector<int>>& stacks_in,
    const std::vector<int>& caps_in,
    int max_iters,
    bool use_deterministic_move
) {
    init_zobrist();

    const int n = (int)stacks_in.size();
    if (n <= 0 || n > MAX_STACKS) throw std::runtime_error("n must be 1..4");
    if ((int)caps_in.size() != n) throw std::runtime_error("caps length mismatch");

    State st;
    st.n = n;

    // caps + check
    for (int i=0;i<n;i++){
        int c = caps_in[i];
        if (c < 0 || c > MAX_CAP) throw std::runtime_error("cap out of range");
        st.cap[i] = (uint8_t)c;
        st.sz[i] = 0;
        for (int p=0;p<MAX_CAP;p++) st.cells[i][p] = EMPTY; // init as empty
    }

    // fill stacks + hash
    st.h = {0,0};
    std::array<int, MAX_TYPES> counts{};
    counts.fill(0);

    for (int s=0;s<n;s++){
        const auto& src = stacks_in[s];
        if ((int)src.size() > (int)st.cap[s]) throw std::runtime_error("initial stack exceeds cap");
        st.sz[s] = (uint8_t)src.size();

        for (int p=0;p<(int)st.cap[s];p++){
            uint8_t v = EMPTY;
            if (p < (int)src.size()) {
                int t = src[p];
                if (t < 0 || t > MAX_TYPES-1) throw std::runtime_error("type out of range");
                v = (uint8_t)t;
                counts[t] += 1;
            }
            st.cells[s][p] = v;
            st.h.a ^= Z1[s][p][v];
            st.h.b ^= Z2[s][p][v];
        }
    }

    // overflow mask (counts are constant across moves)
    st.overflow_mask = 0;
    for (int t=0;t<n;t++){
        if (counts[t] > (int)st.cap[t]) {
            st.overflow_mask |= (uint8_t)(1u << t);
        }
    }

    int threshold2 = heuristic_h2_scaled2(st);
    const int initial_threshold2 = threshold2;

    std::vector<Move> path;
    path.reserve(256);

    bool det = use_deterministic_move;
    int iteration = 0;

    while (iteration < max_iters) {
        iteration++;

        std::unordered_map<Key128, int, KeyHasher> visited;
        visited.reserve(200000); // Adjust as needed (too small causes rehash cost)

        auto res = dfs_ida(st, 0, threshold2, visited, path, det);
        if (res.found) return path;

        if (res.f2 == std::numeric_limits<int>::max()) {
            if (det) {
                // Turn off deterministic and retry
                det = false;
                threshold2 = initial_threshold2;
                path.clear();
                iteration = 0;
                continue;
            }
            throw std::runtime_error("H2: cannot find solution");
        }

        // threshold = max(res_f, threshold + 3.0)  (scaled2 => +6)
        threshold2 = std::max(res.f2, threshold2 + 6);
        path.clear(); // Next iteration starts search from root again
    }

    throw std::runtime_error("H2: max_iters exceeded");
}

PYBIND11_MODULE(cpp_sch, m) {
    m.def(
        "solve_h2_base",
        [](const std::vector<std::vector<int>>& stacks,
           const std::vector<int>& caps,
           int max_iters,
           bool use_deterministic_move) {
            py::gil_scoped_release release;
            return solve_h2_base_cpp(stacks, caps, max_iters, use_deterministic_move);
        },
        py::arg("stacks"),
        py::arg("caps"),
        py::arg("max_iters") = 1000000,
        py::arg("use_deterministic_move") = true
    );
}
