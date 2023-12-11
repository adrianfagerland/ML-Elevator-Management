from abc import ABC, abstractmethod
import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def __init__(self, num_elevators, num_floors, max_speed, max_acceleration) -> None:
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        pass

    @abstractmethod
    def decide(self, observations, error):
        pass
