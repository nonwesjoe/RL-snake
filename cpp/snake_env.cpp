#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <random>

namespace py = pybind11;

struct SnakeEnv {
    int size;
    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;
    std::array<int,2> direction;
    std::vector<std::array<int,2>> snake;
    std::array<int,2> head;
    std::array<int,2> food;
    int steps;
    int score;
    int no_food_steps;
    int last_dist;

    SnakeEnv(int size_=10, py::object seed_obj = py::none()) : size(size_), dist(0, size_-1) {
        if (!seed_obj.is_none()) {
            uint64_t s = seed_obj.cast<uint64_t>();
            rng.seed(s);
        } else {
            std::random_device rd;
            rng.seed(rd());
        }
        reset();
    }

    py::array_t<float> reset() {
        direction = {0, 1};
        int mid = size / 2;
        snake.clear();
        snake.push_back({mid, mid - 1});
        snake.push_back({mid, mid});
        head = snake.back();
        place_food();
        steps = 0;
        score = 0;
        no_food_steps = 0;
        last_dist = manhattan(head, food);
        return get_obs();
    }

    py::tuple step(int action) {
        apply_action(action);
        std::array<int,2> new_head = { head[0] + direction[0], head[1] + direction[1] };
        if (collision(new_head)) {
            py::dict info;
            info["score"] = score;
            return py::make_tuple(get_obs(), -1.0, true, info);
        }
        bool ate = (new_head[0] == food[0] && new_head[1] == food[1]);
        snake.push_back(new_head);
        head = new_head;
        double reward;
        if (ate) {
            score += 1;
            place_food();
            reward = 1.25;
            if (score >= 30) reward += 1.25;
            else if (score >= 20) reward += 1.0;
            no_food_steps = 0;
        } else {
            snake.erase(snake.begin());
            reward = -0.025;
            no_food_steps += 1;
        }
        double turn_pen = (action != 1 ? -0.05 : 0.0);
        int cur_dist = manhattan(head, food);
        double shaping = 0.065 * (static_cast<double>(last_dist - cur_dist));
        last_dist = cur_dist;
        reward = reward + shaping + turn_pen;
        int starve_limit = (size * size) / 6;
        if (!ate && no_food_steps >= starve_limit) {
            py::dict info;
            info["score"] = score;
            steps += 1;
            return py::make_tuple(get_obs(), reward - 0.03, true, info);
        }
        steps += 1;
        py::dict info;
        info["score"] = score;
        return py::make_tuple(get_obs(), reward, false, info);
    }

    void apply_action(int action) {
        if (action == 0) direction = turn_left(direction);
        else if (action == 2) direction = turn_right(direction);
    }

    std::array<int,2> turn_left(const std::array<int,2>& d) const {
        if (d[0] == 0 && d[1] == 1) return {-1, 0};
        if (d[0] == 1 && d[1] == 0) return {0, 1};
        if (d[0] == 0 && d[1] == -1) return {1, 0};
        return {0, -1};
    }

    std::array<int,2> turn_right(const std::array<int,2>& d) const {
        if (d[0] == 0 && d[1] == 1) return {1, 0};
        if (d[0] == 1 && d[1] == 0) return {0, -1};
        if (d[0] == 0 && d[1] == -1) return {-1, 0};
        return {0, 1};
    }

    void place_food() {
        while (true) {
            std::array<int,2> pos = { dist(rng), dist(rng) };
            bool occupied = false;
            for (const auto& seg : snake) {
                if (seg[0] == pos[0] && seg[1] == pos[1]) { occupied = true; break; }
            }
            if (!occupied) { food = pos; break; }
        }
    }

    bool collision(const std::array<int,2>& pos) const {
        if (pos[0] < 0 || pos[0] >= size || pos[1] < 0 || pos[1] >= size) return true;
        for (const auto& seg : snake) {
            if (seg[0] == pos[0] && seg[1] == pos[1]) return true;
        }
        return false;
    }

    int manhattan(const std::array<int,2>& a, const std::array<int,2>& b) const {
        return std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]);
    }

    double free_cells_in_dir(const std::array<int,2>& dvec) const {
        int cnt = 0;
        std::array<int,2> pos = head;
        while (true) {
            pos = { pos[0] + dvec[0], pos[1] + dvec[1] };
            if (collision(pos)) break;
            cnt += 1;
        }
        int denom = std::max(1, size - 1);
        return static_cast<double>(cnt) / static_cast<double>(denom);
    }

    std::vector<float> local_occupancy(int radius=2) const {
        std::array<int,2> f = direction;
        std::array<int,2> l = turn_left(direction);
        int grid_size = 2 * radius + 1;
        std::vector<float> grid(grid_size * grid_size, 0.0f);
        for (int i = -radius; i <= radius; ++i) {
            for (int j = -radius; j <= radius; ++j) {
                if (i == 0 && j == 0) continue;
                std::array<int,2> delta = { i * f[0] + j * l[0], i * f[1] + j * l[1] };
                std::array<int,2> pos = { head[0] + delta[0], head[1] + delta[1] };
                int idx = (i + radius) * grid_size + (j + radius);
                grid[idx] = collision(pos) ? 1.0f : 0.0f;
            }
        }
        return grid;
    }

    py::array_t<float> get_obs() const {
        std::array<int,2> up = {-1, 0};
        std::array<int,2> right = {0, 1};
        std::array<int,2> down = {1, 0};
        std::array<int,2> left = {0, -1};
        float d_up = (direction == up) ? 1.0f : 0.0f;
        float d_right = (direction == right) ? 1.0f : 0.0f;
        float d_down = (direction == down) ? 1.0f : 0.0f;
        float d_left = (direction == left) ? 1.0f : 0.0f;
        double ahead = free_cells_in_dir(direction);
        double leftc = free_cells_in_dir(turn_left(direction));
        double rightc = free_cells_in_dir(turn_right(direction));
        std::array<int,2> fvec = { food[0] - head[0], food[1] - head[1] };
        double norm = static_cast<double>(std::max(1, size - 1));
        double dx = (fvec[0] * direction[0] + fvec[1] * direction[1]) / norm;
        std::array<int,2> ldir = turn_left(direction);
        double dy = (fvec[0] * ldir[0] + fvec[1] * ldir[1]) / norm;
        double length = static_cast<double>(snake.size()) / static_cast<double>(size * size);
        float constant = 1.0f;
        std::vector<float> occ = local_occupancy(2);
        py::array_t<float> obs(occ.size() + 11);
        auto r = obs.mutable_unchecked<1>();
        r(0) = d_up;
        r(1) = d_right;
        r(2) = d_down;
        r(3) = d_left;
        r(4) = static_cast<float>(ahead);
        r(5) = static_cast<float>(leftc);
        r(6) = static_cast<float>(rightc);
        r(7) = static_cast<float>(dx);
        r(8) = static_cast<float>(dy);
        r(9) = static_cast<float>(length);
        r(10) = constant;
        for (size_t i = 0; i < occ.size(); ++i) {
            r(11 + i) = occ[i];
        }
        return obs;
    }

    int get_size() const { return size; }

    std::vector<std::array<int,2>> get_snake() const { return snake; }

    std::array<int,2> get_head() const { return head; }

    std::array<int,2> get_food() const { return food; }

    std::array<int,2> get_direction() const { return direction; }

    void set_direction(std::array<int,2> d) { direction = d; }
};

PYBIND11_MODULE(snake_env_cpp, m) {
    py::class_<SnakeEnv>(m, "SnakeEnv")
        .def(py::init<int, py::object>(), py::arg("size")=10, py::arg("seed")=py::none())
        .def("reset", &SnakeEnv::reset)
        .def("step", &SnakeEnv::step)
        .def("get_size", &SnakeEnv::get_size)
        .def("get_snake", &SnakeEnv::get_snake)
        .def("get_head", &SnakeEnv::get_head)
        .def("get_food", &SnakeEnv::get_food)
        .def("get_direction", &SnakeEnv::get_direction)
        .def("set_direction", &SnakeEnv::set_direction);
}