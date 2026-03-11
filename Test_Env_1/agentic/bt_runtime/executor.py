import py_trees


def execute_tree(
    tree: py_trees.trees.BehaviourTree, max_ticks: int = 10
) -> py_trees.common.Status:
    """Tick a py_trees tree until it reaches a terminal state."""

    for _ in range(max_ticks):
        tree.tick()
        status = tree.root.status
        if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            return status
    raise RuntimeError(f"Tree did not finish within {max_ticks} ticks")
