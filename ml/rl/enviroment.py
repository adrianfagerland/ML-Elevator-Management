import torch

import numpy as np

from tensordict import TensorDict
from torchrl.data import MultiOneHotDiscreteTensorSpec, \
    BinaryDiscreteTensorSpec, \
    CompositeSpec, \
    UnboundedContinuousTensorSpec, \
    BoundedTensorSpec, \
    DiscreteTensorSpec

from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from torchrl.data.utils import DEVICE_TYPING
from torch import device
from matplotlib import pyplot as plt
from elsim.elevator_simulator import ElevatorSimulator

# TODO adjust system enviroment to work with elevator_simulator


class SystemEnvironment(EnvBase):
    def __init__(self, num_elevators, num_floors, device_name="cpu"):
        super(SystemEnvironment, self).__init__()
        self.dtype = np.float32

        self.device = device(device_name)

        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.state_size = self.A.shape[0]
        self.action_size = self.B.shape[1]

        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)

        # Define action specifications
        ac_elevator_target_spec = MultiOneHotDiscreteTensorSpec(nvec=[self.num_floors] * self.num_elevators,
                                                                device=self.device)
        ac_elevator_next_movement_spec = MultiOneHotDiscreteTensorSpec(nvec=[3] * self.num_elevators,
                                                                       device=self.device)

        self.action_spec = CompositeSpec(target=ac_elevator_target_spec, next_movement=ac_elevator_next_movement_spec)

        # Define observation specifications
        ob_el_doors_spec = BoundedTensorSpec(low=0, high=1, shape=torch.Size([self.num_elevators]))
        ob_el_position_speed_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.num_elevators, 2]),)
        ob_el_buttons_spec = BinaryDiscreteTensorSpec(n=self.num_elevators * self.num_floors,
                                                      shape=torch.Size([self.num_elevators, self.num_floors]),
                                                      dtype=torch.bool)
        ob_floors_buttons = BinaryDiscreteTensorSpec(n=2,
                                                     shape=torch.Size([self.num_floors, 2]),
                                                     dtype=torch.bool)
        # maybe TODO add current target as input
        self.observation_spec = CompositeSpec(el_doors=ob_el_doors_spec,
                                              el_pos_speed=ob_el_position_speed_spec,
                                              el_buttons=ob_el_buttons_spec,
                                              fl_buttons=ob_floors_buttons)

        # Define reward specifications (just a real number)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]))

    def _reset(self, tensordict, **kwargs):
        out_tensordict = TensorDict({}, batch_size=torch.Size())

        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)
        out_tensordict.set("observation", torch.tensor(self.state.flatten(), device=self.device))

        return out_tensordict

    def _step(self, tensordict):
        action = tensordict["action"]
        action = action.cpu().numpy().reshape((self.action_size, 1))

        self.state += self.dt * (self.A @ self.state + self.B @ action)

        y = self.C @ self.state + self.D @ action

        error = (self.ref - y) ** 2

        reward = -error

        out_tensordict = TensorDict({"observation": torch.tensor(self.state.astype(self.dtype).flatten(), device=self.device),
                                     "reward": torch.tensor(reward.astype(np.float32), device=self.device),
                                     "done": False}, batch_size=torch.Size())

        return out_tensordict

    def _set_seed(self, seed):
        pass
