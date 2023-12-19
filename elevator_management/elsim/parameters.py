from dataclasses import dataclass
from typing import NewType

# All time values in seconds
DOOR_OPENING_TIME = 2
DOOR_STAYING_OPEN_TIME = 3

FLOOR = NewType("FLOOR", int)
TIME_SEC = NewType("TIME_SEC", float)


DIST_EPSILON = 0.01  # 1/100th of an floor allowed error due to math inaccuracys
LOSS_FACTOR = 10

WAITING_MAX_TIME = 4 * 60  # after 4min a person decides to walk instead of further waiting

REWARD_JOINING_ELEVATOR = 0.2
REWARD_ARRIVE_TARGET = 1
REWARD_CUMMULATIVE_TO_TARGET = 1

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

    time_at_target: float | None = None
    """ time when the person has arrived at its goal, can be None if not yet arrived or never will """


    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
    
    ## Make person hashable
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    def used_stairs(self, world_time) -> bool:
        return world_time - self.original_arrival_time > WAITING_MAX_TIME
    
