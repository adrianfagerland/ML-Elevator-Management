print("TEST STARTED 6", flush=True)


import time
from typing import Type

import gymnasium as gym
import ml.api  # needs to be imported for the env registration
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.env_checker import check_env
from rl.network import (
    OUT_HIDDEN_SIZE,
    PRE_HIDDEN_SIZE,
    ElevatorNetwork,
    alphaLSTMNetwork,
)
from rl.PPO import PPO

env = gym.make("Elevator-v0", num_floors=10, num_elevators=3, num_arrivals=600)

trainer = PPO(alphaLSTMNetwork, env)


trainer.train(save_model="/work/wx350715/elevator_output/model.ml", save_interval=100)

print("TEST WORKED")
