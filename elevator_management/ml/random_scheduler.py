import random

from .scheduler import Scheduler


class RandomScheduler(Scheduler):
    def __init__(self, num_elevators, num_floors, max_speed, max_acceleration) -> None:
        super().__init__(num_elevators, num_floors, max_speed, max_acceleration)

    def decide(self, observations, error):
        targets = [e["target"] for i, e in enumerate(observations["elevators"])]
        random_elevator_index = random.randint(0, self.num_elevators - 1)
        targets[random_elevator_index] = random.randint(0, self.num_floors - 1)

        output = []
        for i in range(self.num_elevators):
            output.append({"target":targets[i], "next_move": random.randint(-1,1)})

        return tuple(output)
