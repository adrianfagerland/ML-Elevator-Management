from dataclasses import dataclass
import numpy as np

# All time values in seconds
DOOR_OPENING_TIME = 2
DOOR_STAYING_OPEN_TIME = 3
INFTY = np.infty

WAITING_MAX_TIME = 4 * 60  # after 4min a person decides to walk instead of further waiting

## Reward parameters
REWARD_JOINING_ELEVATOR = 0.2
""" Rewad for a person entering the elevator"""
REWARD_ARRIVE_TARGET = 1
""" Reward for a person arriving at the target"""
REWARD_OVERTIME_PENALTY_EXP = 4 
"""Higher values lead to more punishment for slower arrival time """
ALLOWED_BASE_TIME = 13
""" Time which is deducted from travel time to account for normal travel time """
ALLOWED_TIME_PER_FLOOR = 5
""" Time which is deducted for each floor travel to account for normal travel time """
TIME_NO_REWRAD = 600
"""Time after which the minimum reward is given"""


REWARD_FORGOT_PEOPLE = -3
""" If people are still in the simulation but all elevators dont have a new target => People would be left => Heavily penalize elevator """
REWARD_NORMALIZER = 1

SIMULATION_END_ARRIVAL_TIME = 300
""" Stores the time after which the elevator simulation stops, after the last person arrived """


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

    time_arrived: float | None = None
    """ time when the person has arrived at its goal or has left because of too long waittime, if not yet left is set to None """

    has_arrived: bool | None = None
    """ Stores if the person has arrived at the target or if it instead has used the stairs."""

    def __lt__(self, other: 'Person'):
        return self.arrival_time < other.arrival_time
    
    
    def __eq__(self, other: 'Person'):
        """ Need to implement an equal function as dataclasses are compared for values. Caused bugs in 'x in self.arrivals' """
        return id(self) == id(other)

    ## Make person hashable
    def get_arrive_reward(self, world_time):
        """ Returns the reward for when arriving at the target. Also sets the internal time_at_target value. 
        Should only be called if actually has arrived """
        self.time_arrived = world_time
        self.has_arrived = True
        # time that is acceptable depending on the travel distance, if used_time is higher penalize reward
        acceptable_time = ALLOWED_BASE_TIME + abs(self.arrival_floor - self. target_floor) * ALLOWED_TIME_PER_FLOOR
        
        travel_time = self.time_arrived - self.original_arrival_time
        # reward factor is a value in (0,1] which determines how much REWARD is actually given
        reward_factor = min(1, ((-travel_time + acceptable_time)/TIME_NO_REWRAD + 1)**REWARD_OVERTIME_PENALTY_EXP)
        return REWARD_ARRIVE_TARGET * reward_factor
    

    def entered_elevator(self, world_time):
        """ Should be called if the person joined an elevator """
        self.entry_elevator_time = world_time
