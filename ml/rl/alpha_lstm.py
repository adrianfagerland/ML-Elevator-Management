import numbers
import warnings
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

    stack_type = StackedLSTM
    layer_type = alpha_LSTMLayer
    dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[alpha_LSTMCell, input_size, hidden_size],
        other_layer_args=[alpha_LSTMCell, hidden_size * dirs, hidden_size],
    )


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


class alpha_LSTMCell(nn.Module):
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


class alpha_LSTMLayer(nn.Module):
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


class StackedLSTM(nn.Module):
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states
