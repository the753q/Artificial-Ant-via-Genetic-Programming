from deap import gp

from shared import (
    AntConfig,
    AntSimulator,
    MAPS_DIR,
    build_pset,
    draw_grid_raylib,
    load_food_grid,
    load_model,
    resolve_model_path,
    run_program,
)

USE_LATEST_EXPERIMENT_MODEL = True
EXPLICIT_EXPERIMENT_NAME = "santa_fe_only"

MAP_PATH = MAPS_DIR / "santa_fe.txt"
VISUALIZE = True


def main() -> None:
    model_path = resolve_model_path(
        use_latest_experiment=USE_LATEST_EXPERIMENT_MODEL,
        explicit_experiment_name=EXPLICIT_EXPERIMENT_NAME,
    )
    model = load_model(model_path)
    grid = load_food_grid(MAP_PATH)
    size = len(grid)

    sim = AntSimulator(
        food_grid=grid,
        config=AntConfig(
            size=size,
            max_steps=model["max_steps_multiplier"] * size * size,
            sensor_range=model["sensor_range"],
            wrap_world=model["wrap_world"],
            pheromone_persistence=model.get("pheromone_persistence", 0.0),
            pheromone_sensor_enabled=model.get(
                "pheromone_sensor_enabled",
                "if_pheromone_ahead" in model["expression"],
            ),
        ),
    )

    pset = build_pset(sim, enabled_sequence_primitives=("prog2", "prog3", "prog4"))
    individual = gp.PrimitiveTree.from_string(model["expression"], pset)
    result = run_program(individual, sim, pset)

    coverage = result["eaten"] / max(1, result["total_food"])
    print("=== TEST RESULT ===")
    print(f"Model: {model_path}")
    print(f"Map: {MAP_PATH}")
    print(f"Grid size: {size}x{size}")
    print(f"Eaten food: {result['eaten']}/{result['total_food']}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Pheromone persistence: {model.get('pheromone_persistence', 0.0):.2f}")
    print(f"Visited cells: {result['visited']}")
    print(f"Steps: {result['steps']}")

    if VISUALIZE:
        draw_grid_raylib(
            size=size,
            base_food_grid=grid,
            history=result["history"],
            max_steps=sim.config.max_steps,
            pheromone_persistence=sim.config.pheromone_persistence,
            pheromone_threshold=sim.config.pheromone_threshold,
        )


if __name__ == "__main__":
    main()
