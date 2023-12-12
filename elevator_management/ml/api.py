import gymnasium as gym
import numpy as np
from elsim.elevator_simulator import ElevatorSimulator

# TODO adjust system enviroment to work with elevator_simulator
from gymnasium import spaces


class ElevatorEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_elevators: tuple[int, int] | int,
        num_floors: int,
        render_mode=None,
        max_speed=2,
        max_acceleration=0.4,
        max_occupancy=7,
    ):
        self.dtype = np.float32
        # Handle the possible two ways to input the parameters of the enviroment: fixed (#elevators/#floors) or a range
        if type(num_elevators) == int:
            assert type(num_elevators) == int
            self.num_elevators_range: tuple[int, int] = (num_elevators, num_elevators + 1)
        else:
            assert type(num_elevators) == tuple[int, int]
            self.num_elevators_range: tuple[int, int] = num_elevators

        self.num_floors = num_floors

        # Parameters that do not change troughout episodes
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_occupancy = max_occupancy

        # To have valid action/observation spaces
        self.reset()

    def reset(self, seed=None, options={}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initializes everything
        self.r = np.random.Generator(np.random.PCG64(seed))

        # 1. choose num_elevators and num_floors for this episode
        self.episode_num_elevators = self.r.integers(*self.num_elevators_range)
        self.episode_num_floors = self.num_floors

        self.simulator: ElevatorSimulator = ElevatorSimulator(
            num_elevators=self.episode_num_elevators,
            num_floors=self.episode_num_floors,
            random_seed=0,
            speed_elevator=self.max_speed,
            acceleration_elevator=self.max_acceleration,
            max_elevator_occupancy=self.max_occupancy,
        )

        # generate the arrival data or read in trough path, TODO: needs to be changed
        self.simulator.init_simulation("data/w1_f9_1.0.1.csv")
        # Define observation space
        self.observation_space = spaces.Dict({
            "floors": spaces.MultiBinary(
                    (self.episode_num_floors, 2), seed=self._get_rnd_int()),
            "num_elevators": spaces.Box(
                low=0, 
                high=self.num_elevators_range[1], 
                shape=(1,), 
                dtype=np.uint8, 
                seed=self._get_rnd_int()),
            "elevators":spaces.Sequence(
                spaces.Dict({    
                    "position": spaces.Box(
                        low=0,
                        high=self.episode_num_floors,
                        shape=(1,),
                        dtype=np.float32,
                        seed=self._get_rnd_int()),
                    "speed": spaces.Box(
                        low=-self.max_speed,
                        high=self.max_speed,
                        shape=(1,),
                        dtype=np.float32,
                        seed=self._get_rnd_int(),
                    ),
                    "buttons": spaces.MultiBinary(
                        (self.episode_num_floors,),
                        seed=self._get_rnd_int(),
                    ),
                    "target": spaces.Discrete(
                        n=self.num_floors,
                        seed=self._get_rnd_int()
                    )
                })
            )
        })


        # Define action space
        self.action_space = spaces.MultiDiscrete(
            [self.episode_num_floors, 3] * self.episode_num_elevators,
            seed=self._get_rnd_int(),
        )
        # Action space cannot be of type dict? for stable baseline3 learning algorithm different shape but contains the same information

        self.action_space = spaces.Dict(spaces={
            "target": spaces.MultiDiscrete([self.episode_num_floors] * self.episode_num_elevators, seed=self._get_rnd_int()),
            "to_serve": spaces.MultiDiscrete([3] * self.episode_num_elevators)
        })

        # return initial observation and info
        observations, _, _, _, info = self.simulator.reset_simulation()
        return (observations, info)

    def _get_rnd_int(self):
        return int(self.r.integers(0, int(1e6)))

    def step(self, action, max_step_size: float | None = None):
        """Function that is called by rollout of the enviroment

        Args:
            tensordict ([tensordict]): [the tensordict that contains the action that should be executed]
            max_step_size: ([float|None]): [If the step size is not none, we limit the max step size of then enviroment
                                        in the info part the algorithm then returns whether an actual decision needs to be done.
                                        If set and no decision is given allows to just continue the simulation with no changes]
        Returns:
            [tensordict]: [the tensordict that contains the observations the reward and if the simulation is finished.]
        """
        # if action is none then pass onwards, ie. no action send to the simulation
        if action is None:
            action_dict = None
        # modify action list to dictionary if not correctly passed as parameter
        # needs to be handled as the policy can only output a list of values while dict is the default for all
        # conventional algorihtms, might not be the best place for this conversion (:shrug)
        elif (not isinstance(action, dict)):
            action_dict = {
                "target": action[::2],
                "to_serve": action[1::2] - 1
            }
        else:

            action_dict = {}
            to_serve_copy = np.copy(action['to_serve']) - 1
            target_copy = np.copy(action['target'])
            action_dict["to_serve"] = to_serve_copy
            action_dict['target'] = target_copy
        # shift the to_serve value to be in range [-1,1] instead of [0,2]

        return self.simulator.step(action_dict, max_step_size=max_step_size)

    def _set_seed(self, seed):
        pass

    def render(self):
        pass


# Register the enviroment


# gymnasium.envs.registration.load_plugin_envs
gym.register(id="Elevator-v0", entry_point="ml.api:ElevatorEnvironment")
