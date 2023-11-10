from ml.api import ElevatorEnvironment
from ml.scheduler import Scheduler


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

    def run(self):
        while not self.done:
            action = self.algorithm.decide(self.observations, self.error)
            self.observations, reward, self.done, self.info = self.api.step(action)
            self.error += reward
        return self.error
