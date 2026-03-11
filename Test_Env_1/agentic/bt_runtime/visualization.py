import py_trees


def render_ascii_tree(tree: py_trees.trees.BehaviourTree) -> str:
    """Return a simple ASCII rendering of a compiled py_trees tree."""

    return py_trees.display.ascii_tree(tree.root)


def print_ascii_tree(tree: py_trees.trees.BehaviourTree) -> None:
    """Print a terminal-friendly rendering of a compiled py_trees tree."""

    print(render_ascii_tree(tree))
