from pathlib import Path

import py_trees


def render_ascii_tree(tree: py_trees.trees.BehaviourTree) -> str:
    """Return a simple ASCII rendering of a compiled py_trees tree."""

    return py_trees.display.ascii_tree(tree.root)


def print_ascii_tree(tree: py_trees.trees.BehaviourTree) -> None:
    """Print a terminal-friendly rendering of a compiled py_trees tree."""

    print(render_ascii_tree(tree))


def export_tree_image(
    tree: py_trees.trees.BehaviourTree,
    name: str = "behavior_tree",
    output_dir: str | Path = "outputs/trees",
) -> dict[str, str] | None:
    """Attempt to export DOT/SVG/PNG artifacts using py_trees built-in rendering."""

    target_directory = Path(output_dir)
    target_directory.mkdir(parents=True, exist_ok=True)
    expected_artifacts = {
        "dot": str(target_directory / f"{name}.dot"),
        "svg": str(target_directory / f"{name}.svg"),
        "png": str(target_directory / f"{name}.png"),
    }

    try:
        rendered = py_trees.display.render_dot_tree(
            tree.root,
            name=name,
            target_directory=str(target_directory),
        )
        artifacts = {key: rendered.get(key, value) for key, value in expected_artifacts.items()}
        print(f"Requested image export: {artifacts['svg']} and {artifacts['png']}")
        return artifacts
    except Exception as exc:
        print(
            "Image export unavailable; text visualization is still available. "
            f"Expected outputs were {expected_artifacts['svg']} and {expected_artifacts['png']}. "
            f"Reason: {exc}"
        )
        return None
