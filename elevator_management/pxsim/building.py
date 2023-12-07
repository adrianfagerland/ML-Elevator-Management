import random
from datetime import datetime, timedelta

import numpy as np

NUM_OF_PEOPLE_PER_FLOOR_PER_ELEVATOR = 50
INTERFLOOR_PROBABILITY = 0.1
PEOPLE_AT_GROUND_FLOOR = 40
WEEKEND_FACTOR = 0.8


class Building:
    def __init__(self, num_of_floors: int, num_of_elevators: int, density: float, seed: int, num_of_people_per_floor_per_elevator: int = NUM_OF_PEOPLE_PER_FLOOR_PER_ELEVATOR, interfloor_probability: float = INTERFLOOR_PROBABILITY, people_at_ground_floor: int = PEOPLE_AT_GROUND_FLOOR, weekend_factor: float = WEEKEND_FACTOR) -> None:
        self.num_of_floors = num_of_floors
        self.num_of_elevators = num_of_elevators
        self.density = density
        self.interfloor_probability = interfloor_probability
        self.r = random.Random(seed)
        self.probability_of_arrival_per_floor_per_person = 0.00000045
        self.weekend_factor = weekend_factor
        self.people_per_floor = [num_of_people_per_floor_per_elevator *
                                 self.num_of_elevators for _ in range(1, num_of_floors + 1)]
        self.people_per_floor.insert(0, people_at_ground_floor)
        seconds_in_a_day = 24 * 60 * 60
        seconds = np.arange(seconds_in_a_day)
        self.distribution = (np.abs(np.sin(2 * np.pi * seconds / seconds_in_a_day - np.pi / 2)) + 0.1) * 100

    def get_next_arrivals(self, time: datetime):
        # determine if the time is a weekend
        if time.weekday() < 5:
            weekend = 0
        else:
            weekend = 1
        current_time = time - timedelta(seconds=1)
        # calculate the number of seconds since start of day
        time_since_start_of_day = (current_time.hour * 60 + current_time.minute) * 60 + current_time.second
        arrivals = []
        while len(arrivals) == 0:
            current_time = current_time + timedelta(seconds=1)
            for floor in range(self.num_of_floors + 1):
                while self.r.random() <= (self.density*self.probability_of_arrival_per_floor_per_person*self.people_per_floor[floor]*self.distribution[time_since_start_of_day]*self.weekend_factor**weekend):
                    if floor == 0 or self.r.random() < self.interfloor_probability:
                        target_floor = self.r.randint(1, self.num_of_floors)
                        while target_floor == floor:
                            target_floor = self.r.randint(1, self.num_of_floors)
                    else:
                        target_floor = 0
                    arrivals.append([time, floor, target_floor])
                    self.people_per_floor[floor] -= 1
                    self.people_per_floor[target_floor] += 1
        return current_time, arrivals
