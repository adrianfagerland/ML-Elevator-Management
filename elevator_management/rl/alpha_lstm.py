from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

"""
Code originally taken from https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py#L32. 
Then modified for our application.
"""


def script_alpha_lstm(
    input_size,
    hidden_size,
    num_layers,
):
    """Returns a ScriptModule for the alpha LSTM module"""

    stack_type = AlphaStackedLSTM
    layer_type = Alpha_LSTMLayer
    dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[Alpha_LSTMCell, input_size, hidden_size],
        other_layer_args=[Alpha_LSTMCell, hidden_size * dirs, hidden_size],
    )


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


class Alpha_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor], alpha: float = 1.0
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        # Modification with alpha (restrict forgetting)
        forgetgate = (alpha * torch.sigmoid(forgetgate)) + (1 - alpha)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        # Modification with alpha II (restrict learning)
        cy = (forgetgate * cx) + alpha * (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class Alpha_LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor], alpha: float = 1.0
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, alpha)
            outputs += [out]
        return torch.stack(outputs), state


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class AlphaStackedLSTM(nn.Module):
    """
    Does yet not work when multiple inputs are given at once.
    """

    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]], alpha: float = 1
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        output_states_h = jit.annotate(List[Tensor], [])
        output_states_c = jit.annotate(List[Tensor], [])
        output = input

        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
            output, out_state = rnn_layer(output, state, alpha)
            output_states_h += [out_state[0].squeeze()]
            output_states_c += [out_state[1].squeeze()]
            i += 1
        return output, (torch.stack(output_states_h), torch.stack(output_states_c))
