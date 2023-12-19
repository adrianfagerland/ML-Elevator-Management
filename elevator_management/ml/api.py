import gymnasium as gym
import numpy as np
from elsim.elevator_simulator import ElevatorSimulator
from ml.feature_extractor import ObservationFeatureExtractor, ActionFeatureExtractor
# TODO adjust system enviroment to work with elevator_simulator
from gymnasium import spaces


class ElevatorEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_elevators: tuple[int, int] | int,
        num_floors: int,
        num_arrivals=2000,
        render_mode=None,
        max_speed=2,
        max_acceleration=0.4,
        max_occupancy=7,
        observation_type: str = "dict",
        action_type: str = "dict"
    ):
        self.dtype = np.float32
        # Handle the possible two ways to input the parameters of the enviroment: fixed (#elevators/#floors) or a range
        if type(num_elevators) == int:
            assert type(num_elevators) == int
            self.num_elevators_range: tuple[int, int] = (
                num_elevators,
                num_elevators + 1,
            )
        else:
            assert type(num_elevators) == tuple[int, int]
            self.num_elevators_range: tuple[int, int] = num_elevators

        self.num_floors = num_floors

        # Defines how the observation space and action space should be treated
        assert observation_type in ['dict', 'discrete']
        self.observation_type = observation_type
        assert action_type in ['dict', 'discrete']
        self.action_type = action_type
        
        # Parameters that do not change troughout episodes
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_occupancy = max_occupancy
        self.num_arrivals = num_arrivals

        # To have valid action/observation spaces
        self.reset()

    def reset(self, seed=None, options={"density":1}):
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
            num_arrivals=self.num_arrivals,
        )

        # generate the arrival data or read in trough path, TODO: needs to be changed
        self.simulator.init_simulation(options['density'])
        # Define observation space
        observation_space_dict = spaces.Dict(
                {
                    "floors": spaces.MultiBinary((self.episode_num_floors, 2), seed=self._get_rnd_int()),
                    "num_elevators": spaces.Box(
                        low=0,
                        high=self.num_elevators_range[1],
                        shape=(1,),
                        dtype=np.uint8,
                        seed=self._get_rnd_int(),
                    ),
                    "time": spaces.Dict(
                        {
                            "time_seconds": spaces.Box(
                                low=0,
                                high=np.infty,
                                shape=(1,),
                                dtype=np.float32,
                                seed=self._get_rnd_int(),
                            ),
                            "time_since_last_seconds": spaces.Box(
                                low=0,
                                high=np.infty,
                                shape=(1,),
                                dtype=np.float32,
                                seed=self._get_rnd_int(),
                            ),
                        }
                    ),
                    "elevators": spaces.Sequence(
                        spaces.Dict(
                            {
                                "position": spaces.Box(
                                    low=0,
                                    high=self.episode_num_floors,
                                    shape=(1,),
                                    dtype=np.float32,
                                    seed=self._get_rnd_int(),
                                ),
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
                                "target": spaces.Discrete(n=self.num_floors, seed=self._get_rnd_int()),
                                "doors_state": spaces.Box(
                                    low=0,
                                    high=1,
                                    shape=(1,),
                                    dtype=np.float32,
                                    seed=self._get_rnd_int(),
                                ),
                                "doors_moving_direction": spaces.Box(low=-1, high=1, shape=(1,), seed=self._get_rnd_int()),
                            }
                        )
                    ),
                }
            )
        if(self.observation_type == 'dict'):
            self.observation_space = observation_space_dict
        elif(self.observation_type == 'discrete'):
            self.observation_feature_extr = ObservationFeatureExtractor(observation_space=observation_space_dict, 
                                                              num_floors=self.num_floors,
                                                              max_elevators=self.num_elevators_range[1] - 1)
            
            self.observation_space = self.observation_feature_extr.return_observation_space

        action_space_dict = spaces.Sequence(
            spaces.Dict({
                "target": spaces.Discrete(self.episode_num_floors, seed=self._get_rnd_int()),
                "next_move": spaces.Discrete(3, start=-1, seed=self._get_rnd_int()),
            })
        )

        if(self.action_type == 'dict'):
            self.action_space = action_space_dict
        elif(self.action_type == 'discrete'):
            self.action_feature_extractor = ActionFeatureExtractor(action_space_dict, 
                                                                   num_floors=self.num_floors, 
                                                                   max_elevators=self.num_elevators_range[1]-1)
            self.action_space = self.action_feature_extractor.return_action_space


        return self.pass_converted_output(self.simulator.reset_simulation())

    def _get_rnd_int(self):
        return int(self.r.integers(0, int(1e6)))

    def step(self, action, max_step_size: float | None = None):
        """Function that is called by rollout of the enviroment

        Args:
            action ([tensordict]): [the tensordict that contains the action that should be executed]
            max_step_size: ([float|None]): [If the step size is not none, we limit the max step size of then enviroment
                                        in the info part the algorithm then returns whether an actual decision needs to be done.
                                        If set and no decision is given allows to just continue the simulation with no changes]
        Returns:
            [tensordict]: [the tensordict that contains the observations the reward and if the simulation is finished.]
        """
        # if action is none then pass onwards, ie. no action send to the simulation
        if action is None:
            action_dict = None
        else:
            # modify action list to dictionary if not correctly passed as parameter
            
            if(self.action_type is dict):
                assert isinstance(action, tuple)
                action_dict = action
            else:
                action_dict = self.action_feature_extractor.extract(action)

        # shift the next_move value to be in range [-1,1] instead of [0,2]
        kwargs = {}
        if max_step_size is not None:
            kwargs["max_step_size"] = max_step_size


        return self.pass_converted_output(self.simulator.step(action_dict, **kwargs))

    def pass_converted_output(self, input):
        """ If the output of simulation needs to be converted from dict to discrete.
            Works as long as the observation is the first entry of input
        """
        if(self.observation_type == 'dict'):
            return input
        
        # Extract observation and info dict
        observation, *rest_info = input
        *rest, info = rest_info
        
        # modify observation
        extracted_obs = self.observation_feature_extr.extract(observation)
        
        info['max_elevators'] = self.observation_feature_extr.max_num_elevators
        info['group_info_len'] = self.observation_feature_extr.group_data_length
        info['elevator_info_len'] = self.observation_feature_extr.elevator_data_length

        return (extracted_obs,) + tuple(rest) + (info,)

    def _set_seed(self, seed):
        pass

    def render(self):
        pass


# Register the enviroment

# gymnasium.envs.registration.load_plugin_envs
gym.register(id="Elevator-v0", entry_point="ml.api:ElevatorEnvironment")
