from enum import Enum, auto


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()
