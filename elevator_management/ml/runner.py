import time

import numpy as np
from ml.api import ElevatorEnvironment
from ml.scheduler import Scheduler
from vis.console import print_elevator


class Runner:
    def __init__(
        self,
        algoritm,
        num_elevators,
        num_floors,
        max_speed=2,
        max_acceleration=0.4,
        seed=0,
    ) -> None:
        self.algorithm: Scheduler = algoritm(
            num_elevators=num_elevators,
            num_floors=num_floors,
            max_speed=max_speed,
            max_acceleration=max_acceleration,
        )
        self.api = ElevatorEnvironment(
            num_elevators=num_elevators,
            num_floors=num_floors,
            max_speed=max_speed,
            max_acceleration=max_acceleration,
        )
        self.error = 0
        self.done = False
        self.truncated = False
        self.needs_decision = True
        self.update_from_observations(self.api.reset(seed=seed))

    def run(self, visualize=False, step_size=0.1):
        # if visualize is True then step size cannot be none
        assert not visualize or step_size is not None
        skipped_time = 0
        if visualize:
            print()
            print_elevator(self.observations, skipped_time, setup=True)
        previous_action = None
        previous_observation = None

        while not self.done:
            if visualize:
                print_elevator(self.observations, skipped_time=skipped_time, previous_action=previous_action)
                if previous_observation is not None and not all(
                    [
                        np.all(previous_observation[key] == self.observations[key])
                        for key in list(self.observations.keys())
                    ]
                ):
                    skipped_time = 0
                else:
                    skipped_time += step_size
                time.sleep(step_size / ((1 + skipped_time) ** 2))
            # If needs decision is true => action to None therefore no elevator will have its target changed
            if self.needs_decision:
                action = self.algorithm.decide(self.observations, self.error)
                previous_action = action
            else:
                action = None
            # If visualize is true then we need to also pass step max_step_size
            previous_observation = self.observations
            self.update_from_observations(
                self.api.step(action, max_step_size=(step_size if visualize else None))
            )

        return self.error

    def update_from_observations(self, observations):
        (
            self.observations,
            reward,
            self.done,
            self.truncated,
            info_dict,
        ) = observations
        self.needs_decision = info_dict['needs_decision']
        self.error += reward
