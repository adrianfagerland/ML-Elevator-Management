from enviroment_gym import ElevatorEnvironment
import gym


gym.register(
    id="Elevator-v0",
    entry_point="enviroment_gym:ElevatorEnvironment",
    kwargs={'num_floors': 10, "num_elevators": 4}
)
env = gym.make('Elevator-v0')
obs = env.reset()
