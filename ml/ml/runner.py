import time

from ml.api import ElevatorEnvironment
from ml.scheduler import Scheduler
from vis.console import print_elevator


class Runner():
    def __init__(self, algoritm, num_elevators, num_floors, max_speed, max_acceleration, seed) -> None:
        self.algorithm: Scheduler = algoritm(
            num_elevators=num_elevators,
            num_floors=num_floors,
            max_speed=max_speed,
            max_acceleration=max_acceleration
        )
        self.api = ElevatorEnvironment(num_elevators=num_elevators,
                                       num_floors=num_floors,
                                       max_speed=max_speed,
                                       max_acceleration=max_acceleration)
        self.observations, self.error, self.done, self.info = self.api.reset(seed=0)

    def run(self, visualize=False):
        while not self.done:
            if visualize:
                print_elevator(self.observations["position"],
                               self.observations["floors"],
                               self.observations["buttons"],
                               self.observations["speed"],
                               self.observations["souls_on_board"])
                time.sleep(0.5)
            action = self.algorithm.decide(self.observations, self.error)
            self.observations, reward, self.done, self.info = self.api.step(action)
            self.error += reward
        return self.error
