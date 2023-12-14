from dataclasses import dataclass
from typing import NewType

# All time values in seconds
DOOR_OPENING_TIME = 2
DOOR_STAYING_OPEN_TIME = 3

FLOOR = NewType("FLOOR", int)
TIME_SEC = NewType("TIME_SEC", float)


@dataclass
class Person:
    original_arrival_time: float
    """ time of arrival"""

    arrival: int
    """the arrival floor"""

    target: int
    """ the target floor"""

    arrival_time: float
    """ time of arrival, can include offset when it had to press the button multiple times"""

    entry_elevator_time: float | None = None
    """ time when entered the elevator, if not in an elevator is set to None """

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time


DIST_EPSILON = 0.01  # 1/100th of an floor allowed error due to math inaccuracys
LOSS_FACTOR = 1e6

WAITING_MAX_TIME = 4 * 60  # after 4min a person decides to walk instead of further waiting
