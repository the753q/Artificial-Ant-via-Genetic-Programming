import csv
import functools
import json
import time
import matplotlib
import multiprocessing
import argparse
import matplotlib.pyplot as plt
from os import cpu_count
from pathlib import Path
from multiprocessing.context import BaseContext
from deap import algorithms, base, creator, gp, tools
from concurrent.futures import ProcessPoolExecutor, as_completed
from shared import (
    AntConfig,
    AntSimulator,
    MAPS_DIR,
    build_pset,
    get_default_model_path,
    get_experiment_dir,
    get_model_dir,
    load_food_grid,
    run_program,
)


matplotlib.use("Agg")

MAP_PATHS = [
    # MAPS_DIR / "basic.txt",
    # MAPS_DIR / "dots.txt",
    # MAPS_DIR / "line.txt",
    # MAPS_DIR / "lines.txt",
    MAPS_DIR / "santa_fe.txt",
    # MAPS_DIR / "spiral.txt",
]

EXPERIMENT_NAME = "santa_fe_only"
MULTI_RUN_ENABLED = True
NUM_RUNS = 15
MAX_PARALLEL_RUNS = cpu_count() or 1

SENSOR_RANGE = 2
WRAP_WORLD = False
PHEROMONE_PERSISTENCE = 0.0
PHEROMONE_SENSOR_ENABLED: bool | None = None
SEQUENCE_PRIMITIVES: tuple[str, ...] = ("prog2", "prog3")
MAX_STEPS_MULTIPLIER = 4

POPULATION = 250
GENERATIONS = 35
TOURNAMENT_SIZE = 5
CX_PROB = 0.6
MUT_PROB = 0.3
MAX_TREE_HEIGHT = 12


def build_train_cases() -> list[dict]:
    train_cases = []
    for map_path in MAP_PATHS:
        grid = load_food_grid(map_path)
        size = len(grid)
        train_cases.append(
            {
                "map_path": map_path.name,
                "grid": grid,
                "size": size,
                "max_steps": MAX_STEPS_MULTIPLIER * size * size,
            }
        )
    return train_cases


def evaluate_individual(individual, train_cases: list[dict]) -> tuple[float]:
    score = 0.0
    for case in train_cases:
        sim = AntSimulator(
            case["grid"],
            AntConfig(
                size=case["size"],
                max_steps=case["max_steps"],
                sensor_range=SENSOR_RANGE,
                wrap_world=WRAP_WORLD,
                pheromone_persistence=PHEROMONE_PERSISTENCE,
                pheromone_sensor_enabled=PHEROMONE_SENSOR_ENABLED,
                record_history=False,
            ),
        )
        result = run_program(individual, sim, build_pset(sim, SEQUENCE_PRIMITIVES))
        score += result["eaten"] / max(1, result["total_food"])

    score -= 0.001 * len(individual)
    return (score,)


def create_toolbox(
    train_cases: list[dict],
) -> tuple[base.Toolbox, gp.PrimitiveSet, functools.partial]:
    first_case = train_cases[0]
    template_sim = AntSimulator(
        first_case["grid"],
        AntConfig(
            size=first_case["size"],
            max_steps=first_case["max_steps"],
            sensor_range=SENSOR_RANGE,
            wrap_world=WRAP_WORLD,
            pheromone_persistence=PHEROMONE_PERSISTENCE,
            pheromone_sensor_enabled=PHEROMONE_SENSOR_ENABLED,
            record_history=False,
        ),
    )
    pset = build_pset(template_sim, SEQUENCE_PRIMITIVES)
    individual_type = getattr(creator, "Individual")
    expr_init = functools.partial(gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    expr_mut = functools.partial(gp.genFull, min_=0, max_=2)
    individual_factory = functools.partial(
        tools.initIterate,
        individual_type,
        expr_init,
    )
    population_factory = functools.partial(tools.initRepeat, list, individual_factory)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate_individual, train_cases=train_cases)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=expr_mut, pset=pset)

    toolbox.decorate(
        "mate", gp.staticLimit(key=lambda ind: ind.height, max_value=MAX_TREE_HEIGHT)
    )
    toolbox.decorate(
        "mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=MAX_TREE_HEIGHT)
    )
    return toolbox, pset, population_factory


def ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=getattr(creator, "FitnessMax"),
        )


def get_multiprocessing_context() -> BaseContext:
    available_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def train_once(run_index: int, train_cases: list[dict], verbose: bool = True) -> dict:
    toolbox, _, population_factory = create_toolbox(train_cases)
    population = population_factory(n=POPULATION)
    hall_of_fame = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats.register("avg", lambda values: sum(values) / len(values))
    stats.register("min", min)
    stats.register("max", max)

    started_at = time.perf_counter()
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=verbose,
    )
    duration_seconds = time.perf_counter() - started_at

    best = hall_of_fame[0]
    return {
        "run_index": run_index,
        "fitness": best.fitness.values[0],
        "tree_size": len(best),
        "duration_seconds": duration_seconds,
        "expression": str(best),
        "model": {
            "expression": str(best),
            "fitness": best.fitness.values[0],
            "sensor_range": SENSOR_RANGE,
            "wrap_world": WRAP_WORLD,
            "pheromone_persistence": PHEROMONE_PERSISTENCE,
            "pheromone_sensor_enabled": (
                PHEROMONE_PERSISTENCE > 0.0
                if PHEROMONE_SENSOR_ENABLED is None
                else PHEROMONE_SENSOR_ENABLED
            ),
            "sequence_primitives": list(SEQUENCE_PRIMITIVES),
            "max_steps_multiplier": MAX_STEPS_MULTIPLIER,
            "train_cases": [
                {"map_path": case["map_path"], "size": case["size"]}
                for case in train_cases
            ],
        },
    }


def train_once_worker(run_index: int, train_cases: list[dict]) -> dict:
    ensure_creator_types()
    return train_once(run_index, train_cases, verbose=False)


def run_training_runs(run_count: int, train_cases: list[dict]) -> list[dict]:
    if run_count == 1:
        print("\n=== TRAIN RUN 1/1 ===")
        return [train_once(1, train_cases, verbose=True)]

    worker_count = min(run_count, MAX_PARALLEL_RUNS)
    print(f"\n=== TRAINING {run_count} RUNS WITH {worker_count} WORKERS ===")

    results_by_index: dict[int, dict] = {}
    completed_count = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_multiprocessing_context(),
    ) as executor:
        future_to_run_index = {
            executor.submit(train_once_worker, run_index, train_cases): run_index
            for run_index in range(1, run_count + 1)
        }
        for future in as_completed(future_to_run_index):
            run_result = future.result()
            results_by_index[run_result["run_index"]] = run_result
            completed_count += 1
            print(
                "Completed run "
                f"{completed_count}/{run_count}: "
                f"fitness={run_result['fitness']:.4f}, "
                f"tree_size={run_result['tree_size']}, "
                f"duration={run_result['duration_seconds']:.2f}s"
            )

    return [results_by_index[index] for index in sorted(results_by_index)]


def write_run_models(model_dir: Path, run_results: list[dict]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    best_run = max(run_results, key=lambda run: run["fitness"])

    default_model_path = get_default_model_path(EXPERIMENT_NAME)
    default_model_path.write_text(
        json.dumps(best_run["model"], indent=2), encoding="utf-8"
    )
    return default_model_path


def write_runs_csv(experiment_dir: Path, run_results: list[dict]) -> Path:
    csv_path = experiment_dir / "run_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run_index",
                "fitness",
                "tree_size",
                "duration_seconds",
                "pheromone_persistence",
                "pheromone_sensor_enabled",
                "sequence_primitives",
                "expression",
            ],
        )
        writer.writeheader()
        for run in run_results:
            writer.writerow(
                {
                    "run_index": run["run_index"],
                    "fitness": f"{run['fitness']:.10f}",
                    "tree_size": run["tree_size"],
                    "duration_seconds": f"{run['duration_seconds']:.4f}",
                    "pheromone_persistence": f"{PHEROMONE_PERSISTENCE:.4f}",
                    "pheromone_sensor_enabled": run["model"][
                        "pheromone_sensor_enabled"
                    ],
                    "sequence_primitives": " ".join(SEQUENCE_PRIMITIVES),
                    "expression": run["expression"],
                }
            )
    return csv_path


def write_boxplot(experiment_dir: Path, run_results: list[dict]) -> Path:
    boxplot_path = experiment_dir / "run_metrics_boxplot.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].boxplot([run["fitness"] for run in run_results], tick_labels=["fitness"])
    axes[0].set_title("Best fitness per run")
    axes[0].set_ylabel("Fitness")

    axes[1].boxplot(
        [run["tree_size"] for run in run_results], tick_labels=["tree_size"]
    )
    axes[1].set_title("Best tree size per run")
    axes[1].set_ylabel("Nodes")

    figure.suptitle(
        f"Experiment: {EXPERIMENT_NAME} | primitives={','.join(SEQUENCE_PRIMITIVES) or 'none'}"
    )
    figure.tight_layout()
    figure.savefig(boxplot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return boxplot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Artificial Ant GP model.")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--max-parallel-runs", type=int, default=MAX_PARALLEL_RUNS)
    parser.add_argument(
        "--pheromone-persistence",
        type=float,
        default=PHEROMONE_PERSISTENCE,
        help="Trail persistence from 0.0 (no trail) to 1.0 (permanent trail).",
    )
    parser.add_argument(
        "--force-pheromone-sensor",
        action="store_true",
        help="Keep if_pheromone_ahead in the GP primitive set even when persistence is 0.0.",
    )
    parser.add_argument(
        "--sequence-primitives",
        nargs="*",
        choices=("prog2", "prog3", "prog4"),
        default=list(SEQUENCE_PRIMITIVES),
        help="Sequence primitives to include in the GP primitive set.",
    )
    return parser.parse_args()


def main() -> None:
    global EXPERIMENT_NAME, NUM_RUNS, MAX_PARALLEL_RUNS, PHEROMONE_PERSISTENCE
    global PHEROMONE_SENSOR_ENABLED, SEQUENCE_PRIMITIVES
    args = parse_args()
    EXPERIMENT_NAME = args.experiment_name
    NUM_RUNS = args.num_runs
    MAX_PARALLEL_RUNS = max(1, args.max_parallel_runs)
    PHEROMONE_PERSISTENCE = max(0.0, min(1.0, args.pheromone_persistence))
    PHEROMONE_SENSOR_ENABLED = (
        True if args.force_pheromone_sensor else PHEROMONE_PERSISTENCE > 0.0
    )
    SEQUENCE_PRIMITIVES = tuple(args.sequence_primitives)

    ensure_creator_types()
    train_cases = build_train_cases()

    run_count = NUM_RUNS if MULTI_RUN_ENABLED else 1
    experiment_dir = get_experiment_dir(EXPERIMENT_NAME)
    model_dir = get_model_dir(EXPERIMENT_NAME)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_results = run_training_runs(run_count, train_cases)

    saved_model_path = write_run_models(model_dir, run_results)
    csv_path = write_runs_csv(experiment_dir, run_results)
    boxplot_path = write_boxplot(experiment_dir, run_results)

    best_run = max(run_results, key=lambda run: run["fitness"])
    print("\n=== TRAIN DONE ===")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Runs: {run_count}")
    print(f"Pheromone persistence: {PHEROMONE_PERSISTENCE:.2f}")
    print(f"Best fitness: {best_run['fitness']:.4f}")
    print(f"Best tree size: {best_run['tree_size']}")
    print(f"Saved best model: {saved_model_path}")
    print(f"Saved run metrics CSV: {csv_path}")
    print(f"Saved boxplot: {boxplot_path}")


if __name__ == "__main__":
    main()
