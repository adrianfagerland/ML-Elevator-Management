from typing import Any

# import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from rl.alpha_lstm import script_alpha_lstm
from rl.feature_extractor import ElevatorFeatureExtractor
from torch import nn
from torch.distributions import Categorical

PRE_HIDDEN_SIZE = 256
COMM_INPUT_SIZE = 128
COMM_HIDDEN_SIZE = 128
OUT_INPUT_SIZE = COMM_HIDDEN_SIZE
OUT_HIDDEN_SIZE = 128


class ElevatorNetwork(nn.Module):
    """A base class for an elevator network. Subclasses should implement the

    Args:
        nn ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_floors: int,
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
        group_info = features[: self.group_data_length]
        num_elevators = int(
            (features.size(dim=0) - self.group_data_length) / self.elevator_data_length
        )
        for ele_idx in range(num_elevators):
            split_features = th.zeros(self.elevator_input_size)
            split_features[: self.group_data_length] = group_info
            feature_tensor = features[
                self.group_data_length
                + ele_idx * self.elevator_data_length : self.group_data_length
                + (ele_idx + 1) * self.elevator_data_length
            ]

            split_features[self.group_data_length :] = feature_tensor
            yield ele_idx, split_features

    def _generate_empty_hidden_state(self):
        raise NotImplementedError("")

    def generate_action_from_output(self, prob):
        # Sample action from output
        target_prob, dir_prob = prob
        distr_target = Categorical(target_prob)
        target = distr_target.sample()
        distr_dir = Categorical(dir_prob)
        direction = distr_dir.sample()

        a = {"target": target, "next_move": direction}
        log_prob_a = (
            distr_target.log_prob(target).sum() + distr_dir.log_prob(direction).sum()
        ).item()
        return a, log_prob_a

    def get_log_prob(self, prob, a):
        # Sample action from output
        if isinstance(prob, list):
            prob_list = []
            for prob_i, a_i in zip(prob, a):
                prob_list.append(self._get_log_prob(prob_i, a_i[0]))
            return th.stack(prob_list)
        else:
            return self._get_log(prob, a)

    def _get_log_prob(self, prob, a):
        target_prob, dir_prob = prob
        distr_target = Categorical(target_prob)
        distr_dir = Categorical(dir_prob)

        log_prob_a = (
            distr_target.log_prob(a["target"]).sum()
            + distr_dir.log_prob(a["next_move"]).sum()
        )
        return log_prob_a


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
        dropoff: float = 0.333,
    ):
        super().__init__(
            observation_space,
            action_space,
            num_floors,
        )

        self.num_rounds = num_rounds
        self.dropoff = dropoff

        self.output_size = num_floors + 3  # this 3 is for up down and stay on floor

        self.setup_preprocess_network()
        self.setup_actor_network()
        self.setup_value_network()

    def setup_actor_network(self):
        # 2 layer LSTM for communication part
        comm_num_layers = 2
        comm_actor_lstm = script_alpha_lstm(
            COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, comm_num_layers
        )

        out_actor_lstm = nn.LSTM(
            self.elevator_input_size + COMM_HIDDEN_SIZE * comm_num_layers,
            OUT_HIDDEN_SIZE,
            1,
        )
        out_actor_fc_tar = nn.Linear(OUT_HIDDEN_SIZE, self.num_floors)
        out_actor_fc_dir = nn.Linear(OUT_HIDDEN_SIZE, 3)
        out_actor_act = nn.Softmax()

        out_actor_target = nn.Sequential(
            out_actor_fc_tar,
            out_actor_act,
        )
        out_actor_dir = nn.Sequential(
            out_actor_fc_dir,
            out_actor_act,
        )
        # create dictionary
        self.actor_layers = nn.ModuleDict(
            {
                "comm": nn.ModuleDict({"lstm": comm_actor_lstm}),
                "out": nn.ModuleDict(
                    {
                        "lstm": out_actor_lstm,
                        "dir": out_actor_dir,
                        "tar": out_actor_target,
                    }
                ),
            }
        )

    def setup_value_network(self):
        # 2 layer LSTM for communication part
        comm_num_layers = 2
        comm_value_lstm = script_alpha_lstm(
            COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, comm_num_layers
        )

        out_value_linear1 = nn.Sequential(
            nn.Linear(
                self.elevator_input_size + COMM_HIDDEN_SIZE * comm_num_layers,
                OUT_HIDDEN_SIZE,
            ),
            nn.LeakyReLU(0.05),
        )

        out_value_linear2 = nn.Linear(OUT_HIDDEN_SIZE, 1)

        # create dictionary
        self.critic_layers = nn.ModuleDict(
            {
                "comm": nn.ModuleDict({"lstm": comm_value_lstm}),
                "out": nn.ModuleDict(
                    {"linear1": out_value_linear1, "linear2": out_value_linear2}
                ),
            }
        )

    def setup_preprocess_network(self):
        # 2 LSTM layer + fully connected as preprocess before communication
        # 2 LSTM layers
        self.preprocess_lstm = nn.LSTM(self.elevator_input_size, PRE_HIDDEN_SIZE, 2)
        # 1 fully connected layer
        self.preprocess_fc = nn.Linear(PRE_HIDDEN_SIZE, COMM_INPUT_SIZE)
        # 1 activation layer
        self.preproces_act = nn.LeakyReLU(0.05)

    def _pre_comm_exec(
        self, split_features, hidden_state: list[tuple], func_set, new_pre_hidden_states
    ):
        preprocessed_input = []
        num_elevators = len(split_features)
        #####################################
        # PREPROCESSING
        #####################################
        for index, split_feature in split_features:
            pre_hidden, _ = hidden_state[index]
            x, new_pre_hidden = self.preprocess_lstm(split_feature[None, :], pre_hidden)
            x = self.preproces_act(self.preprocess_fc(x))

            preprocessed_input.append(x)
            new_pre_hidden_states.append(new_pre_hidden)
        #####################################
        # COMMUNICATION
        #####################################
        comm_hidden = (
            th.zeros([2, COMM_HIDDEN_SIZE], dtype=th.float),
            th.zeros([2, COMM_HIDDEN_SIZE], dtype=th.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            for idx in range(num_elevators):
                # TODO: TWO VERSIONS: USE OR NOT USE OUTPUT
                final_output, comm_hidden = func_set["comm"]["lstm"](
                    preprocessed_input[idx][None, :], comm_hidden, alpha
                )
            alpha *= self.dropoff

        comm_h, comm_c = comm_hidden
        # -> comm_c is the final "decision" that was decided upon (hopefully)
        group_dec_inf = comm_c.flatten()
        return group_dec_inf

    def _forward_single_batch_actor(
        self, features: th.Tensor, hidden_state: list[tuple], func_set
    ) -> tuple[tuple[th.Tensor, th.Tensor], list[tuple]]:
        # to store the new hidden states
        new_pre_hidden_states = []
        new_out_hidden_states = []
        # split up the features
        split_features = list(self.split_features(features))

        finalized_output_tar = []
        finalized_output_dir = []

        group_dec_inf = self._pre_comm_exec(
            split_features, hidden_state, func_set, new_pre_hidden_states
        )

        #####################################
        # POSTPROCESSING
        #####################################
        for index, split_feature in split_features:
            _, out_hidden = hidden_state[index]

            out_input = th.concatenate([group_dec_inf, split_feature])
            out_input, new_out_hidden = func_set["out"]["lstm"](
                out_input[None, :], out_hidden
            )

            out_input_tar = func_set["out"]["tar"](out_input).squeeze()
            out_input_dir = func_set["out"]["dir"](out_input).squeeze()

            finalized_output_tar.append(out_input_tar)
            finalized_output_dir.append(out_input_dir)
            new_out_hidden_states.append(new_out_hidden)

        final_output = (
            th.stack(finalized_output_tar, dim=0),
            th.stack(finalized_output_dir, dim=0),
        )

        return final_output, list(zip(new_pre_hidden_states, new_out_hidden_states))

    def _forward_single_batch_critic(
        self, features: th.Tensor, hidden_state: list[tuple], func_set
    ) -> tuple[th.Tensor, list[tuple]]:
        # to store the new hidden states
        new_pre_hidden_states = []
        # new_out_hidden_states = []
        # split up the features
        split_features = list(self.split_features(features))

        finalized_output_tar = []
        finalized_output_dir = []

        group_dec_inf = self._pre_comm_exec(
            split_features, hidden_state, func_set, new_pre_hidden_states
        )

        #####################################
        # POSTPROCESSIN
        #####################################
        final_output = []
        for index, split_feature in split_features:
            out_input = th.concatenate([group_dec_inf, split_feature])
            out_input = func_set["out"]["linear1"](out_input)
            out_input = func_set["out"]["linear2"](out_input)

            final_output.append(out_input)

        output = th.concatenate(final_output).sum()

        return output, list(
            zip(new_pre_hidden_states, [0] * len(new_pre_hidden_states))
        )

    def _generate_empty_hidden_state(self):
        # structure of hidden_inf_elevator is the following h[0] => contains hidden state of preprocessing => two layers h[0][i] = ith-layer
        #                                                   h[1] => contains hidden state of out layer => one layer => h[1] = 0th layer
        # each layer consist of two values of same number of zeros for both the h and the c value of the lstm
        hidden_inf_one_elevator = (
            th.zeros([2, PRE_HIDDEN_SIZE], dtype=th.float),
            th.zeros([2, PRE_HIDDEN_SIZE], dtype=th.float),
        ), (
            th.zeros([1, OUT_HIDDEN_SIZE], dtype=th.float),
            th.zeros([1, OUT_HIDDEN_SIZE], dtype=th.float),
        )

        return hidden_inf_one_elevator

    def forward_critic(
        self, features: th.Tensor, hidden_state: list[tuple]
    ) -> tuple[th.Tensor, list[tuple]]:
        if features.dim() == 1:  # run just one example
            return self._forward_single_batch_critic(
                features, hidden_state, self.critic_layers
            )
        elif features.dim() == 2:
            # run batch each after another
            out = []
            for single_value in features[:]:
                run_value, hidden_state = self._forward_single_batch_critic(
                    single_value, hidden_state, self.critic_layers
                )
                out.append(run_value)
            return th.stack(out), hidden_state
        else:
            raise Exception()

    def forward_actor(
        self, features: th.Tensor, hidden_state: list[tuple]
    ) -> tuple[tuple[th.Tensor, th.Tensor] | list[tuple], list[tuple]]:
        if features.dim() == 1:  # run just one example
            return self._forward_single_batch_actor(
                features, hidden_state, self.actor_layers
            )
        elif features.dim() == 2:
            # run batch each after another
            out = []
            for single_value in features[:]:
                run_value, hidden_state = self._forward_single_batch_actor(
                    single_value, hidden_state, self.actor_layers
                )
                out.append(run_value)
            return out, hidden_state
        else:
            raise Exception()


class Drop_Arg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor, y) -> th.Tensor:
        return x
