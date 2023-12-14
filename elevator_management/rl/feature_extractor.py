import gymnasium as gym

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.type_aliases import TensorDict
import numpy as np
import torch as th

# from torchrl.data.utils import TensorD
from gymnasium import spaces
from gymnasium.spaces import flatdim, flatten
from torch import nn


# from torchrl.data.utils import TensorD
from gymnasium import spaces
from gymnasium.spaces import flatdim
from torch import nn


class ElevatorFeatureExtractor:
    def __init__(self, observation_space: spaces.Dict, num_floors: int):
        self._observation_space: spaces.Dict = observation_space
        self.num_floors = num_floors

        self.flatten = nn.Flatten()

        # Define the keywords in the observation to expect
        # Figure out the size of data (elevator specific vs group specific)
        # relevant keywords for both data sets
        # data that is specific to an elevator

        self.elevator_keywords = ["position", "speed", "buttons", "target"]

        # and data that is non-specific to an elevator
        self.group_keywords = ["floors"]  # will soon contain more (time, weekday, ...)

        # Figure out when the elevator data starts
        self.group_data_length = 0
        for key in self.group_keywords:
            self.group_data_length += self.flatdim_from_obs_space(observation_space[key])

        self.elevator_data_length = 0

        sequence_space = observation_space["elevators"]
        assert type(sequence_space) == spaces.Sequence
        elevator_observation_space = sequence_space.feature_space
        assert type(elevator_observation_space) == spaces.Dict

        for key in elevator_observation_space:
            self.elevator_data_length += self.flatdim_from_obs_space(elevator_observation_space[key])

    def flatdim_from_obs_space(self, space: spaces.Space) -> int:
        if isinstance(space, spaces.MultiDiscrete):
            return sum(space.nvec)
        elif isinstance(space, spaces.Dict):
            total_size = 0
            for key in space:
                total_size += self.flatdim_from_obs_space(space[key])
                print(key, space[key], total_size)
            return total_size
        else:
            return flatdim(space)

    def flatten_rescale(self, observation, observation_space: spaces.Space) -> np.ndarray:
        if isinstance(observation_space, spaces.Box):
            # Rescale values to be in -1,1
            rg = (observation_space.high + observation_space.low) / 2
            return (observation - rg) / (observation_space.high - rg)
        if isinstance(observation_space, spaces.MultiBinary):
            return observation.flatten()
        if isinstance(observation_space, spaces.Discrete):
            out = np.zeros(observation_space.n)
            out[observation] = 1
            return out
        else:
            return observation.flatten()

    def extract(self, observations) -> th.Tensor:
        # Retrieve num_floors and elevators from observation
        num_floors = observations["floors"].shape[1]
        num_elevators = len(observations["elevators"])

        # Generate empty tensor
        data_out_length = self.group_data_length + num_elevators * self.elevator_data_length
        out_tensor = np.zeros((data_out_length), dtype=np.float32)

        # Fill beginning of empty tensor with the group information
        current_idx = 0
        last_idx = 0
        for key in self.group_keywords:
            current_idx += self.flatdim_from_obs_space(self._observation_space[key])
            out_tensor[last_idx:current_idx] = self.flatten_rescale(observations[key], self._observation_space[key])
            last_idx = current_idx

        sequence_space = self._observation_space["elevators"]
        assert type(sequence_space) == spaces.Sequence
        feature_space = sequence_space.feature_space
        assert type(feature_space) == spaces.Dict

        for ele_idx, elevator_data in enumerate(observations["elevators"]):
            for key in self.elevator_keywords:
                value_size = self.flatdim_from_obs_space(feature_space[key])
                value = self.flatten_rescale(elevator_data[key], feature_space[key])
                out_tensor[last_idx : last_idx + value_size] = value
                last_idx = last_idx + value_size
        return th.Tensor(out_tensor)
