from dataclasses import dataclass
import numpy as np

# All time values in seconds
DOOR_OPENING_TIME = 2
DOOR_STAYING_OPEN_TIME = 3
INFTY = np.infty

WAITING_MAX_TIME = 4 * 60  # after 4min a person decides to walk instead of further waiting

## Reward parameters
REWARD_JOINING_ELEVATOR = 0.2
REWARD_ARRIVE_TARGET = 1
REWARD_OVERTIME_PENALTY_EXP = 2.5  # the higher the more overtime is penalized
REWARD_FORGOT_PEOPLE = -100
REWARD_NORMALIZER = 1


@dataclass
class Person:
    original_arrival_time: float
    """ time of arrival"""

    arrival_floor: int
    """the arrival floor"""

    target_floor: int
    """ the target floor"""

    arrival_time: float
    """ time of arrival, can include offset when it had to press the button multiple times"""

    entry_elevator_time: float | None = None
    """ time when entered the elevator, if not in an elevator is set to None """

    time_left_simulation: float | None = None
    """ time when the person has arrived at its goal or has left because of too long waittime, if not yet left is set to None """

    has_arrived: bool | None = None
    """ Stores if the person has arrived at the target or if it instead has used the stairs."""

    def __lt__(self, other: "Person"):
        return self.arrival_time < other.arrival_time

    def __eq__(self, other: "Person"):
        """Need to implement an equal function as dataclasses are compared for values."""
        return id(self) == id(other)

    ## Make person hashable
    def get_arrive_reward(self, world_time):
        """Returns the reward for when arriving at the target. Also sets the internal time_at_target value.
        Should only be called if actually has arrived"""
        self.time_left_simulation = world_time
        self.has_arrived = True
        # time that is acceptable depending on the travel distance, if used_time is higher penalize reward
        acceptable_time = 10 + abs(self.arrival_floor - self.target_floor) * 3
        travel_time = self.time_left_simulation - self.original_arrival_time

        return REWARD_ARRIVE_TARGET * 1 / max(1, travel_time - acceptable_time) ** REWARD_OVERTIME_PENALTY_EXP

    def get_walk_reward(self, world_time):
        """Returns the (negative) reward for punishing the elevator if a person walked.
        Needs to be tested if a value is even prefered.
        """
        self.time_left_simulation = world_time
        self.has_arrived = False
        return -0

    def entered_elevator(self, world_time):
        """Should be called if the person joined an elevator"""
        self.entry_elevator_time = world_time

    def used_stairs(self, world_time) -> bool:
        return world_time - self.original_arrival_time > WAITING_MAX_TIME
