import torch

import numpy as np

import gym
from gym import spaces
import numpy as np
import pygame


from matplotlib import pyplot as plt
from elsim.elevator_simulator import ElevatorSimulator

# TODO adjust system enviroment to work with elevator_simulator


class ElevatorEnviroment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 num_elevators,
                 num_floors,
                 render_mode=None,
                 max_speed=2,
                 max_acceleration=0.4,
                 seed=0):

        self.dtype = np.float32

        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.simulator = ElevatorSimulator(num_elevators=num_elevators,
                                           num_floors=num_floors,
                                           random_seed=0,
                                           max_speed_elevator=max_speed,
                                           max_acceleration_elevator=max_acceleration)

        self.r = np.random.Generator(np.random.PCG64(seed))

        # Define observation space
        self.observation_space = spaces.Dict({
            "elevator": spaces.Dict({
                "position": spaces.Box(low=0, high=self.num_floors, shape=(self.num_elevators,), dtype=np.float32, seed=self.r.integers(0, int(1e6))),
                "speed": spaces.Box(low=-max_speed, high=max_speed, shape=(self.num_elevators,), dtype=np.float32, seed=self.r.integers(0, int(1e6))),
                "doors_state": spaces.Box(low=0, high=1, shape=(self.num_elevators,), dtype=np.float32, seed=self.r.integers(0, int(1e6))),
                "buttons": spaces.MultiBinary((self.num_elevators, self.num_floors), seed=self.r.integers(0, int(1e6)))
            }),
            "floors": spaces.MultiBinary((self.num_floors, 2), seed=self.r.integers(0, int(1e6)))
        })

        # Define action space
        self.action_space = spaces.Dict({
            "target": spaces.MultiDiscrete([num_floors] * self.num_elevators, seed=self.r.integers(0, int(1e6))),
            "next_movement": spaces.MultiDiscrete([3] * self.num_elevators)
        })

    def _reset(self, tensordict, **kwargs):
        pass

    def step(self, action):
        """ Function that is called by rollout of the enviroment

        Args:
            tensordict ([tensordict]): [the tensordict that contains the action that should be executed]

        Returns:
            [tensordict]: [the tensordict that contains the observations the reward and if the simulation is finished.]
        """

        output = self.simulator.step(action)
        observations = output['observations']
        error = output['error']
        reward = -error
        done = output['done']
        info = None
        out_tuple = (observations, reward, done, info)

        return out_tuple

    def _set_seed(self, seed):
        pass
