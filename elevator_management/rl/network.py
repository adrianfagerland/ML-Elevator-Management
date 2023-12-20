from typing import Any

# import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from ml.scheduler import Scheduler
from rl.alpha_lstm import script_alpha_lstm
from elevator_management.ml.feature_extractor import ObservationFeatureExtractor
from torch import nn
from torch.distributions import Categorical

PRE_HIDDEN_SIZE = 64
COMM_INPUT_SIZE = 64
COMM_HIDDEN_SIZE = 32
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

        self.elevator_input_size = self.group_data_length + self.elevator_data_length

    def sample_action_from_output(self, prob):
        # Sample action from output
        target_prob, dir_prob = th.split(prob, dim=1, split_size_or_sections=[prob.shape[1] - 3, 3])
        distr_target = Categorical(target_prob)
        target = distr_target.sample()
        distr_dir = Categorical(dir_prob)
        direction = distr_dir.sample()

        log_prob_a = (distr_target.log_prob(target).sum() + distr_dir.log_prob(direction).sum()).item()

        # before returning: shift to match real world direction
        direction = direction - 1

        a = {"target": target, "next_move": direction}

        return a, log_prob_a

    def get_log_prob(self, prob, a):
        # Sample action from output
        if isinstance(a, list):
            prob_list = []
            for idx, prob_i in enumerate(prob[:]):
                a_i = a[idx]
                prob_list.append(self._get_log_prob(prob_i, a_i[0]))
            return th.stack(prob_list)
        else:
            return self._get_log_prob(prob, a)

    def _get_log_prob(self, prob, a):
        target_prob, dir_prob = th.split(prob, dim=1, split_size_or_sections=[prob.shape[1] - 3, 3])
        distr_target = Categorical(target_prob)
        distr_dir = Categorical(dir_prob)
        # get indiv actions from a and shift next_move output
        target, next_move = a["target"], a["next_move"]
        next_move += 1

        log_prob_a = distr_target.log_prob(target).sum() + distr_dir.log_prob(next_move).sum()
        return log_prob_a

    def forward_actor(self, features: th.Tensor, hidden_state):
        raise NotImplementedError("Needs to be implemented in subclass")

    def forward_critic(self, features: th.Tensor, hidden_state):
        raise NotImplementedError("Needs to be implemented in subclass")

    def _generate_empty_hidden_state(self):
        raise NotImplementedError("Needs to be implemented in subclass")


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

        self.setup_preprocess_network()
        self.setup_actor_network()
        self.setup_value_network()

    def setup_actor_network(self):
        # 2 layer LSTM for communication part
        comm_num_layers = 2
        comm_actor_lstm = script_alpha_lstm(COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, comm_num_layers)

        out_actor_lstm = nn.LSTM(
            self.elevator_input_size + COMM_HIDDEN_SIZE * comm_num_layers,
            OUT_HIDDEN_SIZE,
            1,
        )
        out_actor_fc_tar = nn.Linear(OUT_HIDDEN_SIZE, self.num_floors)
        out_actor_fc_dir = nn.Linear(OUT_HIDDEN_SIZE, 3)
        out_actor_act = nn.Softmax(dim=1)

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
                "communication_lstm": comm_actor_lstm,
                "postprocessing_lstm": out_actor_lstm,
                "postprocessing_direction": out_actor_dir,
                "postprocessing_target": out_actor_target,
            }
        )

    def setup_value_network(self):
        # define all the layers needed for the critic
        comm_num_layers = 2
        comm_value_lstm = script_alpha_lstm(COMM_INPUT_SIZE, COMM_HIDDEN_SIZE, comm_num_layers)

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
                "communication_lstm": comm_value_lstm,
                "postprocessing_linear1": out_value_linear1,
                "postprocessing_linear2": out_value_linear2,
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

    def _pre_comm_exec(self, split_features, hidden_state: list[tuple], func_set, new_pre_hidden_states):
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
                final_output, comm_hidden = func_set["communication_lstm"](
                    preprocessed_input[idx][None, :], comm_hidden, alpha
                )
            alpha *= self.dropoff

        comm_h, comm_c = comm_hidden
        # -> comm_c is the final "decision" that was decided upon (hopefully)
        group_dec_inf = comm_c.flatten()
        return group_dec_inf

    def _forward_single_batch_actor(
        self, features: th.Tensor, hidden_state: list[tuple]
    ) -> tuple[th.Tensor, list[tuple]]:
        # to store the new hidden states
        new_pre_hidden_states = []
        new_out_hidden_states = []
        # split up the features
        split_features = list(self.split_features(features))

        finalized_output_tar = []
        finalized_output_dir = []

        group_dec_inf = self._pre_comm_exec(split_features, hidden_state, self.actor_layers, new_pre_hidden_states)

        #####################################
        # POSTPROCESSING
        #####################################
        for index, split_feature in split_features:
            _, out_hidden = hidden_state[index]

            out_input = th.concatenate([group_dec_inf, split_feature])
            out_input, new_out_hidden = self.actor_layers["postprocessing_lstm"](out_input[None, :], out_hidden)

            out_input_tar = self.actor_layers["postprocessing_target"](out_input).squeeze()
            out_input_dir = self.actor_layers["postprocessing_direction"](out_input).squeeze()

            finalized_output_tar.append(out_input_tar)
            finalized_output_dir.append(out_input_dir)
            new_out_hidden_states.append(new_out_hidden)

        final_output = (
            th.stack(finalized_output_tar, dim=0),
            th.stack(finalized_output_dir, dim=0),
        )

        return th.concatenate(final_output, dim=1), list(zip(new_pre_hidden_states, new_out_hidden_states))

    def _forward_single_batch_critic(
        self, features: th.Tensor, hidden_state: list[tuple]
    ) -> tuple[th.Tensor, list[tuple]]:
        # to store the new hidden states
        new_pre_hidden_states = []
        # new_out_hidden_states = []
        # split up the features
        split_features = list(self.split_features(features))

        group_dec_inf = self._pre_comm_exec(split_features, hidden_state, self.critic_layers, new_pre_hidden_states)

        #####################################
        # POSTPROCESSIN
        #####################################
        final_output = []
        for index, split_feature in split_features:
            out_input = th.concatenate([group_dec_inf, split_feature])
            out_input = self.critic_layers["postprocessing_linear1"](out_input)
            out_input = self.critic_layers["postprocessing_linear2"](out_input)

            final_output.append(out_input)

        output = th.concatenate(final_output).sum()

        return output, list(zip(new_pre_hidden_states, [0] * len(new_pre_hidden_states)))

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

    def _forward(self, features: th.Tensor, hidden_state: list[tuple], func_single_batch):
        if features.dim() == 1:  # run just one example
            return func_single_batch(features, hidden_state)
        elif features.dim() == 2:
            # run batch each after another
            out = []
            for single_value in features[:]:
                run_value, hidden_state = func_single_batch(single_value, hidden_state)
                out.append(run_value)
            return th.stack(out), hidden_state
        else:
            raise Exception()

    def forward_actor(self, features: th.Tensor, hidden_state: list[tuple]):
        return self._forward(features, hidden_state, self._forward_single_batch_actor)

    def forward_critic(self, features: th.Tensor, hidden_state: list[tuple]):
        return self._forward(features, hidden_state, self._forward_single_batch_critic)


"""
    def forward(self, features, hidden_states):
        hidden_actor, hidden_critic = hidden_states
        return self.forward_actor(features, hidden_actor), self.forward_critic(features, hidden_critic)
"""
