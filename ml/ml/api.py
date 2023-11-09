import gymnasium as gym
import numpy as np

from elsim.elevator_simulator import ElevatorSimulator

# TODO adjust system enviroment to work with elevator_simulator

# gym.register(
#     id="Elevator-v0",
#     entry_point="enviroment_gym:ElevatorEnvironment",
#     kwargs={'num_floors': 10, "num_elevators": 4}
# )
# env = gym.make('Elevator-v0')
# obs = env.reset()


class ElevatorEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 num_elevators: tuple[int, int] | int,
                 num_floors: tuple[int, int] | int,
                 render_mode=None,
                 max_speed=2,
                 max_acceleration=0.4):

        self.dtype = np.float32
        # Handle the possible two ways to input the parameters of the enviroment: fixed (#elevators/#floors) or a range
        if (type(num_elevators) == int):
            self.num_elevators_range = [num_elevators, num_elevators + 1]
        else:
            self.num_elevators_range = num_elevators

        if (type(num_floors) == int):
            self.num_floors_range = [num_floors, num_floors + 1]
        else:
            self.num_floors_range = num_floors


        # Parameters that do not change troughout episodes
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration

    def reset(self, seed = 0) -> tuple:
        self.r = np.random.Generator(np.random.PCG64(seed))
        # Initializes everything

        # 1. choose num_elevators and num_floors for this episode
        self.episode_num_elevators = self.r.integers(*self.num_elevators_range)
        self.episode_num_floors = self.r.integers(*self.num_floors_range)

        self.simulator = ElevatorSimulator(num_elevators=self.episode_num_elevators,
                                           num_floors=self.episode_num_floors,
                                           random_seed=0,
                                           speed_elevator=self.max_speed,
                                           acceleration_elevator=self.max_acceleration)

        # generate the arrival data or read in trough path
        self.simulator.init_simulation("../pxsim/data/test_data.csv")

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "elevator": gym.spaces.Dict({
                "position": gym.spaces.Box(low=0, high=self.episode_num_floors, shape=(self.episode_num_elevators,), dtype=np.float32, seed=self._get_rnd_int()),
                "speed": gym.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(self.episode_num_elevators,), dtype=np.float32, seed=self._get_rnd_int()),
                "doors_state": gym.spaces.Box(low=0, high=1, shape=(self.episode_num_elevators,), dtype=np.float32, seed=self._get_rnd_int()),
                "buttons": gym.spaces.MultiBinary((self.episode_num_elevators, self.episode_num_floors), seed=self._get_rnd_int())
            }),
            "floors": gym.spaces.MultiBinary((self.episode_num_floors, 2), seed=self._get_rnd_int())
        })

        # Define action space
        self.action_space = gym.spaces.Dict({
            "target": gym.spaces.MultiDiscrete([self.episode_num_floors] * self.episode_num_elevators, seed=self._get_rnd_int()),
            "next_movement": gym.spaces.MultiDiscrete([3] * self.episode_num_elevators)
        })

        # TODO return initial observation
        return (None, None, None, None)

    def _get_rnd_int(self):
        return int(self.r.integers(0, int(1e6)))

    def step(self, action):
        """ Function that is called by rollout of the enviroment

        Args:
            tensordict ([tensordict]): [the tensordict that contains the action that should be executed]

        Returns:
            [tensordict]: [the tensordict that contains the observations the reward and if the simulation is finished.]
        """
        # modify action (next_movement) from [0,2] to [-1,1]
        action['next_movement'] -= 1
        output = self.simulator.step(action)
        observations = output['observations']
        error = output['loss']
        reward = -error
        done = output['done']
        info = None
        out_tuple = (observations, reward, done, info)

        return out_tuple

    def _set_seed(self, seed):
        pass