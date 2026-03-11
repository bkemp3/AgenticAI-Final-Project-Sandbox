from pathlib import Path
import sys

# Allow running the demo directly from the repository root with uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agent import Agent
from agentic.world_state import WorldState


def main() -> None:
    world_state = WorldState(
        robot_position="home",
        object_visible=False,
        holding_object=False,
        target_object="cube",
    )
    agent = Agent(world_state)
    structure = agent.plan("pickup_object")

    print("Structured behavior tree:")
    print(structure.model_dump_json(indent=2))

    agent.run_structure(structure)


if __name__ == "__main__":
    main()
