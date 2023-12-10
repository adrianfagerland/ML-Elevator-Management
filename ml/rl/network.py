from typing import Any
from gymnasium import spaces
# import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from rl.alpha_lstm import script_alpha_lstm
from rl.feature_extractor import ElevatorFeatureExtractor



PRE_HIDDEN_SIZE = 256
COMM_INPUT_SIZE = 128
COMM_HIDDEN_SIZE = 128
OUT_INPUT_SIZE = COMM_HIDDEN_SIZE
OUT_HIDDEN_SIZE = 128


class ElevatorNetwork(nn.Module):
    """ A base class for an elevator network. Subclasses should implement the 

    Args:
        nn ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_floors: int
    ):
        super().__init__()

        self.num_floors = num_floors
        # dirty hack: we cant assume observation_space to be dict in the type hint as
        # this would throw an error as the return general return type of the env is spaces.Space

        assert type(observation_space) == spaces.Dict
        # Elevator Feature Extractor: should be called before trying to analyse the input data
        # Reorders the data and makes it NN friendly
        self.feature_extractor = ElevatorFeatureExtractor(observation_space, num_floors)

        self.elevator_data_length = self.feature_extractor.elevator_data_length
        self.group_data_length = self.feature_extractor.group_data_length

        self.elevator_input_size = self.group_data_length + self.elevator_data_length

    def forward_actor(self, features: th.Tensor, hidden_state):
        raise NotImplementedError("Needs to be implemented in subclass")

    def forward_critic(self, features: th.Tensor, hidden_state):
        raise NotImplementedError("Needs to be implemented in subclass")

    def extract_features(self, features: th.Tensor) -> th.Tensor:
        return self.feature_extractor.extract(features)

    def split_features(self, features: th.Tensor):
        group_info = features[:self.group_data_length]
        num_elevators = int((features.size(dim=0) - self.group_data_length) / self.elevator_data_length)
        for ele_idx in range(num_elevators):
            split_features = th.zeros(self.elevator_input_size)
            split_features[:self.group_data_length] = group_info
            feature_tensor = features[self.group_data_length+ele_idx *
                                      self.elevator_data_length:self.group_data_length+(ele_idx+1)*self.elevator_data_length]

            split_features[self.group_data_length:] = feature_tensor
            yield ele_idx, split_features

    def generate_empty_hidden_state(self):
        raise NotImplementedError("")


class alphaLSTMNetwork(ElevatorNetwork):
    """
    Custom network for policy and value function that implements the alpha-LSTM approach.
    It receives as input the features extracted by the features extractor.

    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        num_floors: int,
        num_rounds: int = 3,
        dropoff: float = 0.333
    ):
        super().__init__(observation_space, action_space, num_floors, )

        self.num_rounds = num_rounds
        self.dropoff = dropoff

        self.setup_preprocess_network()
        self.setup_actor_network()
        self.setup_value_network()

    def setup_actor_network(self):
        # 2 layer LSTM for communication part
        self.comm_actor_lstm = script_alpha_lstm(COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, 2)

        self.out_actor_lstm = nn.LSTM(self.elevator_input_size + COMM_HIDDEN_SIZE, OUT_HIDDEN_SIZE, 1)
        self.out_actor_fc = nn.Linear(OUT_HIDDEN_SIZE, self.num_floors)

    def setup_value_network(self):
        # 2 layer LSTM for communication part
        self.comm_value_lstm = nn.LSTM(COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, 2)

        # fc layer for after
        self.comm_value_fc = nn.Linear(COMM_HIDDEN_SIZE, OUT_INPUT_SIZE)

        self.out_value_lstm = nn.LSTM(OUT_INPUT_SIZE, OUT_HIDDEN_SIZE, 1)
        self.out_value_fc = nn.Linear(OUT_HIDDEN_SIZE, self.num_floors)

    def setup_preprocess_network(self):
        # 2 LSTM layer + fully connected as preprocess before communication
        # 2 LSTM layers
        self.preprocess_lstm = nn.LSTM(self.elevator_input_size, PRE_HIDDEN_SIZE, 2)
        # 1 fully connected layer
        self.preprocess_fc = nn.Linear(PRE_HIDDEN_SIZE, COMM_INPUT_SIZE)
        # 1 activation layer
        self.preproces_act = nn.LeakyReLU(0.05)

    def forward_actor(self, features: th.Tensor, hidden_state: tuple[th.Tensor, th.Tensor]) -> tuple[th.Tensor, tuple]:

        # to store the new hidden states
        new_pre_hidden_states = []
        new_out_hidden_states = []
        # split up the features
        split_features = list(self.split_features(features))
        num_elevators = len(split_features)

        preprocessed_input = []
        finalized_output = []

        for index, split_feature in split_features:
            pre_hidden, _ = hidden_state[index]
            x, new_pre_hidden = self.preprocess_lstm(split_feature, pre_hidden)
            x = self.preproces_act(self.preprocess_fc(x))

            preprocessed_input.append(x)
            new_pre_hidden_states.append(new_pre_hidden)

        # communication round
        comm_hidden = (th.zeros([1, 1, COMM_HIDDEN_SIZE], dtype=th.float),
                       th.zeros([1, 1, COMM_HIDDEN_SIZE], dtype=th.float))
        for num_rounds in range(self.num_rounds):
            alpha = 1
            for idx in range(num_elevators):
                # TWO VERSIONS: USE OR NOT USE OUTPUT
                output, comm_hidden = self.comm_actor_lstm(preprocessed_input[idx], comm_hidden, alpha)
            alpha *= self.dropoff

        comm_h, comm_c = comm_hidden
        # -> comm_c is the final "decision" that was decided upon (hopefully)

        for index, split_feature in split_features:
            _, out_hidden = hidden_state[index]

            out_input = th.stack([comm_c, split_feature])
            out_input, new_out_hidden = self.out_actor_lstm(out_input, out_hidden)

            out_input = self.out_actor_fc(out_input)

            finalized_output.append(out_input)
            new_out_hidden_states.append(new_out_hidden)

        return th.stack(finalized_output), (tuple(new_pre_hidden_states), tuple(new_out_hidden_states))

    def generate_empty_hidden_state(self):
        # structure of hidden_inf_elevator is the following h[0] => contains hidden state of preprocessing => two layers h[0][i] = ith-layer
        #                                                   h[1] => contains hidden state of out layer => one layer => h[1] = 0th layer
        # each layer consist of two values of same number of zeros for both the h and the c value of the lstm

        hidden_inf_one_elevator = ((
            (th.zeros([1, 1, PRE_HIDDEN_SIZE], dtype=th.float),
                th.zeros([1, 1, PRE_HIDDEN_SIZE], dtype=th.float)),
            (th.zeros([1, 1, PRE_HIDDEN_SIZE], dtype=th.float),
                th.zeros([1, 1, PRE_HIDDEN_SIZE], dtype=th.float))
        ),
            (th.zeros([1, 1, OUT_HIDDEN_SIZE], dtype=th.float), th.zeros([1, 1, OUT_HIDDEN_SIZE], dtype=th.float)))

        return hidden_inf_one_elevator

    def forward_critic(self, features: th.Tensor, hidden_state: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        raise NotImplementedError("Needs to be implemented in subclass")


pass
