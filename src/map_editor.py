from pathlib import Path

import raylib as rl

from shared import MAPS_DIR


DEFAULT_GRID_SIZE = 32
MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 128
MIN_CELL_SIZE = 8
MAX_CELL_SIZE = 28
TOOLBAR_HEIGHT = 88
PADDING = 16
FPS = 60


def _rl_fn(snake: str, camel: str):
    fn = getattr(rl, snake, None)
    if fn is None:
        fn = getattr(rl, camel, None)
    if fn is None:
        raise AttributeError(f"raylib binding missing both '{snake}' and '{camel}'")
    return fn


def _call_with_text_fallback(fn, *args):
    try:
        return fn(*args)
    except TypeError:
        converted = [a.encode("utf-8") if isinstance(a, str) else a for a in args]
        return fn(*converted)


def empty_grid(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def load_grid(path: Path) -> list[list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Map file is empty: {path}")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"Map must be rectangular: {path}")
    if len(rows) != len(rows[0]):
        raise ValueError(f"Map must be square: {path}")

    allowed = {"-": 0, "0": 1}
    try:
        return [[allowed[cell] for cell in row] for row in rows]
    except KeyError as exc:
        raise ValueError(f"Unsupported map character: {exc.args[0]!r}") from None


def save_grid(path: Path, grid: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join("".join("0" if cell else "-" for cell in row) for row in grid)
    path.write_text(content + "\n", encoding="utf-8")


def clear_grid(grid: list[list[int]]) -> None:
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            grid[y][x] = 0


def clamp_grid_size(size: int) -> int:
    return max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, size))


def map_path_from_name(name: str) -> Path:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Enter a file name first")
    if trimmed.endswith(".txt"):
        trimmed = trimmed[:-4]
    return MAPS_DIR / f"{trimmed}.txt"


def point_in_rect(point, rect) -> bool:
    x, y, width, height = rect
    return x <= point.x <= x + width and y <= point.y <= y + height


def draw_button(
    draw_rectangle,
    draw_rectangle_lines,
    draw_text,
    get_mouse_position,
    rect,
    label,
):
    x, y, width, height = rect
    mouse = get_mouse_position()
    hovered = point_in_rect(mouse, rect)
    bg = rl.LIGHTGRAY if hovered else rl.RAYWHITE
    draw_rectangle(x, y, width, height, bg)
    draw_rectangle_lines(x, y, width, height, rl.GRAY)
    _call_with_text_fallback(
        draw_text,
        label,
        int(x + 12),
        int(y + 10),
        20,
        rl.BLACK,
    )


def draw_textbox(
    draw_rectangle,
    draw_rectangle_lines,
    draw_text,
    rect,
    value: str,
    focused: bool,
    placeholder: str,
):
    x, y, width, height = rect
    border = rl.MAROON if focused else rl.GRAY
    draw_rectangle(x, y, width, height, rl.RAYWHITE)
    draw_rectangle_lines(x, y, width, height, border)
    if focused:
        draw_rectangle_lines(x + 1, y + 1, width - 2, height - 2, border)
    text = value or placeholder
    color = rl.BLACK if value else rl.GRAY
    _call_with_text_fallback(
        draw_text,
        text,
        int(x + 10),
        int(y + 10),
        20,
        color,
    )


def append_text(current: str, char_code: int, max_length: int) -> str:
    if len(current) >= max_length:
        return current
    if 32 <= char_code <= 126:
        return current + chr(char_code)
    return current


def main() -> None:
    grid_size = DEFAULT_GRID_SIZE
    grid = empty_grid(grid_size)
    file_name = ""
    grid_size_text = str(grid_size)
    status = "New empty grid"
    focused_field = None

    set_config_flags = _rl_fn("set_config_flags", "SetConfigFlags")
    init_window = _rl_fn("init_window", "InitWindow")
    set_target_fps = _rl_fn("set_target_fps", "SetTargetFPS")
    window_should_close = _rl_fn("window_should_close", "WindowShouldClose")
    begin_drawing = _rl_fn("begin_drawing", "BeginDrawing")
    clear_background = _rl_fn("clear_background", "ClearBackground")
    draw_rectangle = _rl_fn("draw_rectangle", "DrawRectangle")
    draw_rectangle_lines = _rl_fn("draw_rectangle_lines", "DrawRectangleLines")
    draw_text = _rl_fn("draw_text", "DrawText")
    end_drawing = _rl_fn("end_drawing", "EndDrawing")
    close_window = _rl_fn("close_window", "CloseWindow")
    get_mouse_position = _rl_fn("get_mouse_position", "GetMousePosition")
    is_mouse_button_down = _rl_fn("is_mouse_button_down", "IsMouseButtonDown")
    is_mouse_button_pressed = _rl_fn("is_mouse_button_pressed", "IsMouseButtonPressed")
    is_key_pressed = _rl_fn("is_key_pressed", "IsKeyPressed")
    get_char_pressed = _rl_fn("get_char_pressed", "GetCharPressed")
    get_screen_width = _rl_fn("get_screen_width", "GetScreenWidth")
    get_screen_height = _rl_fn("get_screen_height", "GetScreenHeight")

    width = 980
    height = 860
    set_config_flags(rl.FLAG_WINDOW_RESIZABLE)
    _call_with_text_fallback(init_window, width, height, "Map editor")
    set_target_fps(FPS)
    last_painted = None

    while not window_should_close():
        screen_width = get_screen_width()
        screen_height = get_screen_height()
        canvas_top = TOOLBAR_HEIGHT + PADDING
        canvas_left = PADDING
        canvas_width = max(200, screen_width - 2 * PADDING)
        canvas_height = max(200, screen_height - canvas_top - PADDING)
        cell_size = max(
            MIN_CELL_SIZE,
            min(
                MAX_CELL_SIZE,
                min(canvas_width // grid_size, canvas_height // grid_size),
            ),
        )
        grid_pixel_size = grid_size * cell_size
        grid_origin_x = canvas_left + max(0, (canvas_width - grid_pixel_size) // 2)
        grid_origin_y = canvas_top + max(0, (canvas_height - grid_pixel_size) // 2)

        file_box = (PADDING + 64, 20, 260, 40)
        load_button = (PADDING + 336, 20, 86, 40)
        save_button = (PADDING + 432, 20, 86, 40)
        clear_button = (PADDING + 528, 20, 86, 40)
        size_box = (PADDING + 712, 20, 72, 40)
        new_button = (PADDING + 794, 20, 86, 40)
        status_rect = (PADDING + 64, 64, screen_width - 2 * PADDING - 64, 20)

        mouse = get_mouse_position()
        inside = (
            grid_origin_x <= mouse.x < grid_origin_x + grid_pixel_size
            and grid_origin_y <= mouse.y < grid_origin_y + grid_pixel_size
        )
        grid_x = int((mouse.x - grid_origin_x) // cell_size) if inside else -1
        grid_y = int((mouse.y - grid_origin_y) // cell_size) if inside else -1

        if is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
            if point_in_rect(mouse, file_box):
                focused_field = "file"
            elif point_in_rect(mouse, size_box):
                focused_field = "size"
            else:
                focused_field = None

            if point_in_rect(mouse, load_button):
                try:
                    path = map_path_from_name(file_name)
                    grid = load_grid(path)
                    grid_size = len(grid)
                    grid_size_text = str(grid_size)
                    status = f"Loaded {path.name}"
                except (FileNotFoundError, ValueError) as exc:
                    status = str(exc)

            elif point_in_rect(mouse, save_button):
                try:
                    path = map_path_from_name(file_name)
                    save_grid(path, grid)
                    status = f"Saved {path.name}"
                except ValueError as exc:
                    status = str(exc)

            elif point_in_rect(mouse, clear_button):
                clear_grid(grid)
                status = "Cleared grid"

            elif point_in_rect(mouse, new_button):
                try:
                    requested_size = clamp_grid_size(int(grid_size_text))
                    grid_size = requested_size
                    grid_size_text = str(requested_size)
                    grid = empty_grid(grid_size)
                    status = f"Created new {grid_size}x{grid_size} grid"
                except ValueError:
                    status = "Grid size must be a number"

        if focused_field is not None:
            char_code = get_char_pressed()
            while char_code > 0:
                if focused_field == "file":
                    file_name = append_text(file_name, char_code, 40)
                elif focused_field == "size":
                    if chr(char_code).isdigit() and len(grid_size_text) < 3:
                        grid_size_text += chr(char_code)
                char_code = get_char_pressed()

            if is_key_pressed(rl.KEY_BACKSPACE):
                if focused_field == "file":
                    file_name = file_name[:-1]
                elif focused_field == "size":
                    grid_size_text = grid_size_text[:-1]

        if not (
            is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
            or is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT)
        ):
            last_painted = None

        if inside:
            if is_mouse_button_down(rl.MOUSE_BUTTON_LEFT):
                if last_painted != (grid_x, grid_y, 1):
                    grid[grid_y][grid_x] = 1
                    last_painted = (grid_x, grid_y, 1)
            elif is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT):
                if last_painted != (grid_x, grid_y, 0):
                    grid[grid_y][grid_x] = 0
                    last_painted = (grid_x, grid_y, 0)

        if is_key_pressed(rl.KEY_C) and focused_field is None:
            clear_grid(grid)
            status = "Cleared grid"
        if is_key_pressed(rl.KEY_S) and focused_field is None:
            try:
                path = map_path_from_name(file_name)
                save_grid(path, grid)
                status = f"Saved {path.name}"
            except ValueError as exc:
                status = str(exc)

        begin_drawing()
        clear_background(rl.RAYWHITE)

        _call_with_text_fallback(draw_text, "File", PADDING, 30, 20, rl.BLACK)
        _call_with_text_fallback(draw_text, "Grid", 662, 30, 20, rl.BLACK)
        draw_textbox(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            file_box,
            file_name,
            focused_field == "file",
            "map name",
        )
        draw_textbox(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            size_box,
            grid_size_text,
            focused_field == "size",
            str(DEFAULT_GRID_SIZE),
        )
        draw_button(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            get_mouse_position,
            load_button,
            "Load",
        )
        draw_button(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            get_mouse_position,
            save_button,
            "Save",
        )
        draw_button(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            get_mouse_position,
            clear_button,
            "Clear",
        )
        draw_button(
            draw_rectangle,
            draw_rectangle_lines,
            draw_text,
            get_mouse_position,
            new_button,
            "New",
        )
        _call_with_text_fallback(
            draw_text,
            "Type file.txt or file. Load/Save always uses root/data/maps. Left drag paints 0, right drag erases to -.",
            int(status_rect[0]),
            int(status_rect[1]),
            18,
            rl.DARKGRAY,
        )
        _call_with_text_fallback(
            draw_text,
            status,
            int(status_rect[0]),
            int(status_rect[1] + 20),
            18,
            rl.MAROON,
        )

        for y in range(grid_size):
            for x in range(grid_size):
                px = grid_origin_x + x * cell_size
                py = grid_origin_y + y * cell_size
                color = rl.RED if grid[y][x] else rl.WHITE
                draw_rectangle(px, py, cell_size, cell_size, color)
                draw_rectangle_lines(px, py, cell_size, cell_size, rl.LIGHTGRAY)

        end_drawing()

    close_window()


if __name__ == "__main__":
    main()
