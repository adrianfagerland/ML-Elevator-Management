import time

import numpy as np
from ml.api import ElevatorEnvironment
from ml.scheduler import Scheduler
from vis.console import ConsoleVisualizer
from vis.plot import PyGameVisualizer
from vis.visualizer import Visualizer


class Runner:
    def __init__(
        self,
        algoritm,
        num_elevators,
        num_floors,
        max_speed=2,
        max_acceleration=0.4,
        seed=0,
<<<<<<< Updated upstream
        should_plot=False,
        visualizer="console",
=======
<<<<<<< Updated upstream
=======
        visualizer: str | None = "console",
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    ) -> None:
        assert visualizer in ["console", "pygame"]
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
        self.should_plot = should_plot
        obs, info = self.api.reset(seed=seed)
        self.elapsed_time = 0
        self.update_from_observations(obs, info_dict=info)
<<<<<<< Updated upstream
        self.visualizer: Visualizer = ConsoleVisualizer() if visualizer == "console" else PyGameVisualizer(obs)
=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes

    def run(self, visualize=False, step_size=0.1):
        skipped_time = 0
<<<<<<< Updated upstream

=======
=======
        if visualizer is not None:
            assert visualizer in ["console", "pygame"]
            self.visualizer: Visualizer = ConsoleVisualizer() if visualizer == "console" else PyGameVisualizer(obs)

    def run(self, visualize=False, step_size=0.1):
        assert step_size > 0
        assert not (visualize and self.visualizer is None)

>>>>>>> Stashed changes
>>>>>>> Stashed changes
        if visualize:
            self.visualizer.setup()

        previous_action = None
        time_last_print = time.time()

        while not self.done:
            # If needs decision is true => action to None therefore no elevator will have its target changed
            if self.needs_decision:
                action = self.algorithm.decide(self.observations, self.error)
                previous_action = action
            else:
                action = None

            # If visualize is true then we need to also pass step max_step_size
            obs, reward, done, trunc, info = self.api.step(action, max_step_size=(step_size if visualize else None))
            self.update_from_observations(obs, reward=reward, done=done, trunc=trunc, info_dict=info)

            if visualize:
                time_since_last_print = time.time() - time_last_print
                self.visualizer.visualize(self.observations, previous_action=previous_action)
                time.sleep(max(step_size - time_since_last_print, 0))
                time_last_print = time.time()

        return self.error

    def update_from_observations(self, obs, info_dict, reward=0, done=False, trunc=False):
        # store data for next run
        self.observations = obs
        self.done = done
        self.truncated = trunc
        self.needs_decision = info_dict["needs_decision"]
        # log data
        self.error += reward
