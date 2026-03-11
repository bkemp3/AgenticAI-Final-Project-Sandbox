from agentic.bt_runtime.leaf_behaviors import (
    DetectObjectBehaviour,
    HoldingObjectBehaviour,
    ObjectVisibleBehaviour,
    PickObjectBehaviour,
)


LEAF_BEHAVIOUR_REGISTRY = {
    "detect_object": DetectObjectBehaviour,
    "pick_object": PickObjectBehaviour,
    "object_visible": ObjectVisibleBehaviour,
    "holding_object": HoldingObjectBehaviour,
}
