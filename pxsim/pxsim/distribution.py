from typing import List
import numpy as np


class Distribution:
    def __init__(self) -> None:
        self.second = 24 * 60 * 60
        self.X = np.linspace(0, self.second, self.second)
        self.P = [self.dist_function(x) for x in self.X]
        self.P = np.array(self.P)
        self.P = self.P / np.sum(self.P)

    def generate_distribution(self, num_of_entries: int) -> List[float]:
        """Generates which second someone appears to want the elevator based on a complicated distribution function `dist_function()`.

        Args:
            num_of_entries (int): How many entries should there be every day

        Yields:
            Generator[int, None, None]: return the second of the day when someone wants to enter the elevator
        """
        entries = []
        for _ in range(num_of_entries):
            entries.append(np.random.choice(self.X, p=self.P))
        return entries

    def dist_function(self, x) -> float:
        return np.abs(np.sin(2 * np.pi * x / self.second - np.pi / 2)) + 0.1
