import argparse
from pathlib import Path
import sys

# Allow running the demo directly from the repository root with uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.simulation import CollectionWorld, load_collection_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "collection_env.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the timed value-collection simulator demo.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the collection environment YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, objects = load_collection_config(args.config)
    world = CollectionWorld(config=config, objects=objects)

    print(f"Loaded config from: {args.config}")
    print(f"Config: {config}")
    print("Initial partial observation (radius=1, hidden object details):")
    print(world.get_observation(visibility_radius=1, include_object_details=False))

    print("Advancing the world by one timestep without an action:")
    world.step()
    print(world.get_observation(visibility_radius=1, include_object_details=False))

    actions = [
        ("move", (1, 0)),
        ("pickup", "bronze_cube"),
        ("move", (0, 0)),
        ("deposit", None),
        ("move", (1, 0)),
        ("move", (2, 0)),
        ("pickup", "silver_cylinder"),
        ("move", (1, 0)),
        ("move", (0, 0)),
        ("deposit", None),
    ]

    for action, payload in actions:
        if world.is_terminal():
            break
        if action == "move":
            result = world.move_robot(payload)
        elif action == "pickup":
            result = world.attempt_pickup(payload)
        elif action == "deposit":
            result = world.deposit_object()
        else:
            raise ValueError(f"Unsupported action: {action}")
        print(f"Action {action}({payload}) -> {result}")
        print(f"Observation: {world.get_observation(visibility_radius=1, include_object_details=False)}")

    print("Final full observation:")
    print(world.get_observation())
    print(f"Final metrics: {world.get_metrics()}")
    print("Event log:")
    for event in world.events:
        print(f"  {event}")


if __name__ == "__main__":
    main()
