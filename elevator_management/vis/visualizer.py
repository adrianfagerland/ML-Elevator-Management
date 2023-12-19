from abc import ABC, abstractmethod


class Visualizer(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def visualize(self, observations, previous_action):
        pass
