import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ml.api import ElevatorEnvironment

env = gym.make('Elevator-v0', num_floors=20, num_elevators=5)
check_env(env)

model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10)
