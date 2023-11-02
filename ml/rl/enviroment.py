import torch

import numpy as np

from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

from matplotlib import pyplot as plt


# TODO adjust system enviroment to work with elevator_simulator



class SystemEnvironment(EnvBase):
    def __init__(self, A, B, C, D, dt, ref=1, device="cpu"):
        super(SystemEnvironment, self).__init__()
        self.dtype = np.float32

        self.A, self.B, self.C, self.D, self.dt, self.ref = A, B, C, D, dt, ref
        self.device = device

        self.state_size = self.A.shape[0]
        self.action_size = self.B.shape[1]

        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)

        self.action_spec = BoundedTensorSpec(minimum=-1, maximum=1, shape=torch.Size([self.action_size]))

        observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.state_size]))
        self.observation_spec = CompositeSpec(observation=observation_spec)

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