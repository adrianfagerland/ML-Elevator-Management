from typing import Type
import datetime

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
import os
from pathlib import Path

# Create Log Path
log_folder = Path(os.getcwd())
# Check if in correct folder (might not be neccessary idk)
if(os.path.isdir(log_folder/"elevator_management")):
    log_folder = log_folder/"elevator_management"

log_folder = log_folder/"logs"

env = gym.make("Elevator-v0", num_floors=10, num_elevators=3, num_arrivals=100)
# check_env(env.unwrapped)

trainer = PPO(alphaLSTMNetwork, env, log_folder=log_folder)

trainer.train(save_interval=100)
