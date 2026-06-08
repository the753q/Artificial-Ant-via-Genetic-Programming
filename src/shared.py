import json
from functools import partial
from pathlib import Path
from typing import Callable
from deap import gp


DIRECTIONS = [
    (0, -1),  # hore
    (1, 0),  # vpravo
    (0, 1),  # dole
    (-1, 0),  # vlavo
]


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MAPS_DIR = DATA_DIR / "maps"
RESULTS_DIR = BASE_DIR.parent / "results"


class AntConfig:
    def __init__(
        self,
        size: int,
        max_steps: int,
        sensor_range: int = 1,
        wrap_world: bool = True,
        pheromone_persistence: float = 0.0,
        pheromone_threshold: float = 0.05,
        pheromone_sensor_enabled: bool | None = None,
        record_history: bool = True,
    ):
        self.size = size
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.wrap_world = wrap_world
        self.pheromone_persistence = max(0.0, min(1.0, pheromone_persistence))
        self.pheromone_threshold = pheromone_threshold
        self.pheromone_sensor_enabled = (
            self.pheromone_persistence > 0.0
            if pheromone_sensor_enabled is None
            else pheromone_sensor_enabled
        )
        self.record_history = record_history


class AntSimulator:
    visited: set[tuple[int, int]] | None
    path: list[tuple[int, int]] | None
    history: list[dict] | None

    def __init__(self, food_grid: list[list[int]], config: AntConfig):
        self.original_food = [row[:] for row in food_grid]
        self.config = config
        self.total_food = sum(sum(row) for row in self.original_food)
        self.reset()

    def reset(self) -> None:
        self.food = [row[:] for row in self.original_food]
        self.x = 0
        self.y = 0
        self.dir_idx = 1  # zacina doprava
        self.steps = 0
        self.eaten = 0
        self.visited = {(0, 0)} if self.config.record_history else None
        self.path = [(0, 0)] if self.config.record_history else None
        self.pheromone_step = [
            [-1 for _ in range(self.config.size)] for _ in range(self.config.size)
        ]
        if self.config.pheromone_persistence > 0.0:
            self.pheromone_step[0][0] = 0
        # Eat food at starting position if it exists
        if self.food[0][0] == 1:
            self.food[0][0] = 0
            self.eaten += 1
        self.history = [self._snapshot("start")] if self.config.record_history else None

    def visible_cells(self) -> list[tuple[int, int]]:
        dx, dy = DIRECTIONS[self.dir_idx]
        x, y = self.x, self.y
        cells = []
        for _ in range(self.config.sensor_range):
            x, y = self._normalize(x + dx, y + dy)
            cells.append((x, y))
        return cells

    def _snapshot(self, action: str) -> dict:
        return {
            "action": action,
            "x": self.x,
            "y": self.y,
            "dir_idx": self.dir_idx,
            "step": self.steps,
            "visible": self.visible_cells(),
        }

    def _normalize(self, x: int, y: int) -> tuple[int, int]:
        n = self.config.size
        if self.config.wrap_world:
            return x % n, y % n
        return max(0, min(n - 1, x)), max(0, min(n - 1, y))

    def food_ahead(self) -> bool:
        dx, dy = DIRECTIONS[self.dir_idx]
        x, y = self.x, self.y
        for _ in range(self.config.sensor_range):
            x, y = self._normalize(x + dx, y + dy)
            if self.food[y][x] == 1:
                return True
        return False

    def pheromone_ahead(self) -> bool:
        persistence = self.config.pheromone_persistence
        if persistence <= 0.0:
            return False

        dx, dy = DIRECTIONS[self.dir_idx]
        x, y = self.x, self.y
        for _ in range(self.config.sensor_range):
            x, y = self._normalize(x + dx, y + dy)
            last_step = self.pheromone_step[y][x]
            if last_step < 0:
                continue
            if persistence >= 1.0:
                return True
            age = self.steps - last_step
            if persistence**age >= self.config.pheromone_threshold:
                return True
        return False

    def update_pheromone(self) -> None:
        if self.config.pheromone_persistence > 0.0:
            self.pheromone_step[self.y][self.x] = self.steps

    def move_forward(self) -> None:
        if self.steps >= self.config.max_steps or self.eaten >= self.total_food:
            return
        dx, dy = DIRECTIONS[self.dir_idx]
        self.x, self.y = self._normalize(self.x + dx, self.y + dy)
        self.steps += 1
        if self.config.record_history:
            assert self.path is not None and self.visited is not None
            self.path.append((self.x, self.y))
            self.visited.add((self.x, self.y))
        self.update_pheromone()
        if self.food[self.y][self.x] == 1:
            self.food[self.y][self.x] = 0
            self.eaten += 1
        if self.config.record_history:
            assert self.history is not None
            self.history.append(self._snapshot("move_forward"))

    def turn_left(self) -> None:
        if self.steps >= self.config.max_steps or self.eaten >= self.total_food:
            return
        self.dir_idx = (self.dir_idx - 1) % 4
        self.steps += 1
        self.update_pheromone()
        if self.config.record_history:
            assert self.history is not None
            self.history.append(self._snapshot("turn_left"))

    def turn_right(self) -> None:
        if self.steps >= self.config.max_steps or self.eaten >= self.total_food:
            return
        self.dir_idx = (self.dir_idx + 1) % 4
        self.steps += 1
        self.update_pheromone()
        if self.config.record_history:
            assert self.history is not None
            self.history.append(self._snapshot("turn_right"))

    def if_food_ahead(
        self, out1: Callable[[], None], out2: Callable[[], None]
    ) -> Callable[[], None]:
        return partial(_if_then_else, self.food_ahead, out1, out2)

    def if_pheromone_ahead(
        self, out1: Callable[[], None], out2: Callable[[], None]
    ) -> Callable[[], None]:
        return partial(_if_then_else, self.pheromone_ahead, out1, out2)


def _if_then_else(
    condition: Callable[[], bool], out1: Callable[[], None], out2: Callable[[], None]
) -> None:
    if condition():
        out1()
    else:
        out2()


def _progn(*funcs: Callable[[], None]) -> None:
    for fn in funcs:
        fn()


def prog2(out1: Callable[[], None], out2: Callable[[], None]) -> Callable[[], None]:
    return partial(_progn, out1, out2)


def prog3(
    out1: Callable[[], None], out2: Callable[[], None], out3: Callable[[], None]
) -> Callable[[], None]:
    return partial(_progn, out1, out2, out3)


def prog4(
    out1: Callable[[], None],
    out2: Callable[[], None],
    out3: Callable[[], None],
    out4: Callable[[], None],
) -> Callable[[], None]:
    return partial(_progn, out1, out2, out3, out4)


def build_pset(
    simulator: AntSimulator,
    enabled_sequence_primitives: tuple[str, ...] = ("prog2", "prog3"),
) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(simulator.if_food_ahead, 2)
    if simulator.config.pheromone_sensor_enabled:
        pset.addPrimitive(simulator.if_pheromone_ahead, 2)
    if "prog2" in enabled_sequence_primitives:
        pset.addPrimitive(prog2, 2)
    if "prog3" in enabled_sequence_primitives:
        pset.addPrimitive(prog3, 3)
    if "prog4" in enabled_sequence_primitives:
        pset.addPrimitive(prog4, 4)
    pset.addTerminal(simulator.move_forward)
    pset.addTerminal(simulator.turn_left)
    pset.addTerminal(simulator.turn_right)
    return pset


def run_program(individual, simulator: AntSimulator, pset: gp.PrimitiveSet) -> dict:
    simulator.reset()
    routine = gp.compile(individual, pset)
    while (
        simulator.steps < simulator.config.max_steps
        and simulator.eaten < simulator.total_food
    ):
        routine()
    return {
        "eaten": simulator.eaten,
        "steps": simulator.steps,
        "visited": len(simulator.visited) if simulator.visited is not None else 0,
        "path": simulator.path[:] if simulator.path is not None else [],
        "history": simulator.history[:] if simulator.history is not None else [],
        "total_food": sum(sum(r) for r in simulator.original_food),
    }


def load_food_grid(path: str | Path) -> list[list[int]]:
    map_path = Path(path)
    rows = [
        line.strip()
        for line in map_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Map file is empty: {map_path}")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Map must be rectangular: {map_path}")
    if width != len(rows):
        raise ValueError(f"Map must be square for this simulator: {map_path}")

    allowed = {"-": 0, "0": 1}
    grid: list[list[int]] = []
    for y, row in enumerate(rows, start=1):
        try:
            grid.append([allowed[cell] for cell in row])
        except KeyError as exc:
            bad = exc.args[0]
            raise ValueError(
                f"Unsupported map character {bad!r} at line {y} in {map_path}"
            ) from None

    return grid


def get_experiment_dir(experiment_name: str) -> Path:
    return RESULTS_DIR / experiment_name


def get_model_dir(experiment_name: str) -> Path:
    return get_experiment_dir(experiment_name) / "ant_model"


def get_default_model_path(experiment_name: str) -> Path:
    return get_model_dir(experiment_name) / "ant_model.json"


def iter_experiment_model_paths() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []

    # Build list of paths that exist
    existing_paths = [
        candidate / "ant_model" / "ant_model.json"
        for candidate in RESULTS_DIR.iterdir()
        if candidate.is_dir() and (candidate / "ant_model" / "ant_model.json").exists()
    ]

    # Sort by modification time
    return sorted(
        existing_paths,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_model_path(
    use_latest_experiment: bool, explicit_experiment_name: str | None = None
) -> Path:
    # Prioritize explicit experiment name if provided
    if explicit_experiment_name:
        model_path = get_default_model_path(explicit_experiment_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    # Fall back to latest experiment if no explicit name given
    if use_latest_experiment:
        model_paths = [path for path in iter_experiment_model_paths() if path.exists()]
        if not model_paths:
            raise FileNotFoundError(
                f"No trained models found in results directory: {RESULTS_DIR}"
            )
        return model_paths[0]

    # Neither explicit name nor use_latest enabled
    raise ValueError(
        "Either explicit_experiment_name must be provided or use_latest_experiment must be True."
    )


def load_model(path: str | Path) -> dict:
    model_path = Path(path)
    return json.loads(model_path.read_text(encoding="utf-8"))


def draw_grid_raylib(
    size: int,
    base_food_grid: list[list[int]],
    path: list[tuple[int, int]] | None = None,
    history: list[dict] | None = None,
    max_steps: int | None = None,
    pheromone_persistence: float = 0.0,
    pheromone_threshold: float = 0.05,
    fps: int = 24,
    cell_size: int = 24,
    margin: int = 40,
) -> None:
    import raylib as rl

    def _call_with_text_fallback(fn, *args):
        try:
            return fn(*args)
        except TypeError:
            converted = [a.encode("utf-8") if isinstance(a, str) else a for a in args]
            return fn(*converted)

    def _rl_fn(snake: str, camel: str):
        fn = getattr(rl, snake, None)
        if fn is None:
            fn = getattr(rl, camel, None)
        if fn is None:
            raise AttributeError(f"raylib binding missing both '{snake}' and '{camel}'")
        return fn

    color_ctor = getattr(rl, "Color", None) or getattr(rl, "color", None)
    if color_ctor is not None:
        eaten_food_color = color_ctor(255, 0, 0, 40)
    else:
        fade = getattr(rl, "fade", None) or getattr(rl, "Fade", None)
        if fade is not None:
            eaten_food_color = fade(rl.RED, 0.15)
        else:
            get_color = getattr(rl, "get_color", None) or getattr(rl, "GetColor", None)
            eaten_food_color = (
                get_color(0xFF000028) if get_color is not None else rl.LIGHTGRAY
            )

    init_window = _rl_fn("init_window", "InitWindow")
    set_target_fps = _rl_fn("set_target_fps", "SetTargetFPS")
    window_should_close = _rl_fn("window_should_close", "WindowShouldClose")
    begin_drawing = _rl_fn("begin_drawing", "BeginDrawing")
    clear_background = _rl_fn("clear_background", "ClearBackground")
    draw_rectangle = _rl_fn("draw_rectangle", "DrawRectangle")
    draw_circle = _rl_fn("draw_circle", "DrawCircle")
    draw_circle_lines = _rl_fn("draw_circle_lines", "DrawCircleLines")
    draw_triangle = _rl_fn("draw_triangle", "DrawTriangle")
    draw_text = _rl_fn("draw_text", "DrawText")
    end_drawing = _rl_fn("end_drawing", "EndDrawing")
    close_window = _rl_fn("close_window", "CloseWindow")
    get_time = _rl_fn("get_time", "GetTime")
    is_key_down = _rl_fn("is_key_down", "IsKeyDown")
    vector_ctor = getattr(rl, "Vector2", None) or getattr(rl, "vector2", None)

    def _vec(x: float, y: float):
        return vector_ctor(x, y) if vector_ctor is not None else (x, y)

    def _draw_thick_circle_lines(x: int, y: int, radius: int, color) -> None:
        for offset in range(3):
            draw_circle_lines(x, y, radius - offset, color)

    def _draw_bold_text(text: str, x: int, y: int, font_size: int, color) -> None:
        for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            _call_with_text_fallback(draw_text, text, x + ox, y + oy, font_size, color)

    def _build_pheromone_steps(current_index: int) -> list[list[int]]:
        pheromone_steps = [[-1 for _ in range(size)] for _ in range(size)]
        if pheromone_persistence <= 0.0:
            return pheromone_steps
        assert history is not None
        for snapshot in history[: current_index + 1]:
            pheromone_steps[snapshot["y"]][snapshot["x"]] = snapshot["step"]
        return pheromone_steps

    def _pheromone_digit(last_step: int, current_step: int) -> int:
        if last_step < 0:
            return 0
        if pheromone_persistence >= 1.0:
            return 9
        age = max(0, current_step - last_step)
        strength = pheromone_persistence**age
        if strength < pheromone_threshold:
            return 0
        return max(1, min(9, int(strength * 9 + 0.999999)))

    set_config_flags = getattr(rl, "set_config_flags", None) or getattr(
        rl, "SetConfigFlags", None
    )
    if set_config_flags is not None:
        set_config_flags(rl.FLAG_WINDOW_RESIZABLE)

    get_screen_width = _rl_fn("get_screen_width", "GetScreenWidth")
    get_screen_height = _rl_fn("get_screen_height", "GetScreenHeight")

    if history is None:
        history = [
            {
                "action": "move_forward",
                "x": x,
                "y": y,
                "dir_idx": 1,
                "step": index,
                "visible": [],
            }
            for index, (x, y) in enumerate(path or [(0, 0)])
        ]

    info_height = 40
    width = size * cell_size + 2 * margin
    height = size * cell_size + 2 * margin + info_height

    _call_with_text_fallback(init_window, width, height, "Artificial Ant - replay")
    set_target_fps(fps)

    idx = 0
    consumed = set()
    paused = True
    space_was_down = False
    space_pressed_at: float | None = None
    tap_threshold = 0.18
    normal_interval = 1.0 / max(1, fps)
    slow_interval = normal_interval * 4
    display_max_steps = max_steps if max_steps is not None else len(history) - 1
    next_advance_time = get_time() + normal_interval
    while not window_should_close():
        now = get_time()
        space_is_down = is_key_down(rl.KEY_SPACE)
        if space_is_down and not space_was_down:
            space_pressed_at = now
        elif not space_is_down and space_was_down:
            if (
                space_pressed_at is not None
                and (now - space_pressed_at) < tap_threshold
            ):
                paused = not paused
            space_pressed_at = None
        space_was_down = space_is_down

        slow_mode = (
            space_is_down
            and space_pressed_at is not None
            and (now - space_pressed_at) >= tap_threshold
        )
        advance_interval = slow_interval if slow_mode else normal_interval

        screen_width = get_screen_width()
        screen_height = get_screen_height()
        available_width = max(1, screen_width - 2 * margin)
        available_height = max(1, screen_height - 2 * margin - info_height)
        cell_size_px = max(1, min(available_width // size, available_height // size))
        grid_pixel_size = size * cell_size_px
        grid_origin_x = max(margin, (screen_width - grid_pixel_size) // 2)
        grid_origin_y = max(
            margin, (screen_height - info_height - grid_pixel_size) // 2
        )

        if (
            not paused
            and idx < len(history) - 1
            and now >= next_advance_time
        ):
            idx += 1
            next_advance_time = now + advance_interval
        elif paused:
            next_advance_time = now + advance_interval

        current_index = min(idx, len(history) - 1)
        state = history[current_index]
        ax, ay = state["x"], state["y"]
        if base_food_grid[ay][ax] == 1:
            consumed.add((ax, ay))
        pheromone_steps = _build_pheromone_steps(current_index)

        begin_drawing()
        clear_background(rl.RAYWHITE)

        current = state
        ant_x, ant_y = current["x"], current["y"]

        for y in range(size):
            for x in range(size):
                px = grid_origin_x + x * cell_size_px
                py = grid_origin_y + y * cell_size_px

                color = rl.LIGHTGRAY
                if base_food_grid[y][x] == 1:
                    color = eaten_food_color if (x, y) in consumed else rl.RED
                if (ant_x, ant_y) == (x, y):
                    color = rl.BLACK

                draw_rectangle(px, py, cell_size_px - 1, cell_size_px - 1, color)

                if (x, y) in current["visible"] and (x, y) != (ant_x, ant_y):
                    cx = px + cell_size_px // 2
                    cy = py + cell_size_px // 2
                    draw_circle(cx, cy, max(4, cell_size_px // 5), rl.WHITE)

                if (x, y) != (ant_x, ant_y):
                    digit = _pheromone_digit(
                        pheromone_steps[y][x], current["step"]
                    )
                    if digit > 0:
                        font_size = max(12, int(cell_size_px * 0.72))
                        text = str(digit)
                        text_x = px + max(1, (cell_size_px - font_size // 2) // 2)
                        text_y = py + max(1, (cell_size_px - font_size) // 2)
                        _draw_bold_text(text, text_x, text_y, font_size, rl.BLACK)

        center_x = grid_origin_x + ant_x * cell_size_px + cell_size_px // 2
        center_y = grid_origin_y + ant_y * cell_size_px + cell_size_px // 2
        radius = max(5, cell_size_px // 3)
        dx, dy = DIRECTIONS[current["dir_idx"]]
        tip = _vec(center_x + dx * radius, center_y + dy * radius)
        left = _vec(center_x - dy * radius * 0.7, center_y + dx * radius * 0.7)
        right = _vec(center_x + dy * radius * 0.7, center_y - dx * radius * 0.7)
        draw_circle(center_x, center_y, max(3, cell_size_px // 5), rl.BLACK)
        draw_triangle(tip, left, right, rl.WHITE)

        action_no = current_index
        info = f"steps: {action_no}/{display_max_steps}"
        _call_with_text_fallback(
            draw_text, info, margin, screen_height - 30, 20, rl.BLACK
        )

        end_drawing()

    close_window()
