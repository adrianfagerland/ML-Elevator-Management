import time

from ml.api import ElevatorEnvironment
from ml.nearest_car import NearestCar
from ml.scheduler import Scheduler
from vis.console import print_elevator


class Runner():
    def __init__(self, algoritm, num_elevators, num_floors, max_speed=2, max_acceleration=0.4, seed=0) -> None:
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
        self.observations, self.info = self.api.reset(seed=0)
        self.error = 0
        self.done = False
        self.truncated = False
        self.needs_decision = True

    def run(self, visualize=False, step_size=0.1):

        # if visualize is True then step size cannot be none
        assert not visualize or step_size is not None
        door_state = self.observations['doors_state']
        print_elevator(self.observations, door_state, setup=True)
        previous_action = None

        while not self.done:
            if visualize:
                print_elevator(self.observations, door_state, previous_action)
                door_state = self.observations['doors_state']

                time.sleep(step_size)
            # If needs decision is true => action to None therefore no elevator will have its target changed
            if (self.needs_decision):
                action = self.algorithm.decide(self.observations, self.error)
                previous_action = action
            else:
                action = None
            # If visualize is true then we need to also pass step max_step_size
            if (visualize):
                self.observations, reward, self.done, self.truncated, self.info = self.api.step(
                    action, max_step_size=step_size)
            else:
                self.observations, reward, self.done, self.truncated, self.info = self.api.step(action)

            self.needs_decision = self.info["needs_decision"]
            self.error += reward

        return self.error


if __name__ == "__main__":
    r = Runner(NearestCar, 5, 10, seed=0)
    r.run(visualize=True)
