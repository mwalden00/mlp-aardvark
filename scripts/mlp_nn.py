import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

torch.manual_seed(1234567891)
random.seed(1234567891)


class base_LSTM(nn.Module):
    """Base LSTM class (stcked LSTM with Linear Activation at end)"""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_lstm_layers,
        out_dim,
        bias=False,
        dropout=0.0,
        bi_d=False,
        batch_first=True,
        ens_model=None,
    ):
        """
        @params
        - int in_dim: input feature dimensionality
        - int hidden_dim: number hidden nodes
        - int num_lstm_layers: number of LSTM layers
        - in out_dim: output feature dimensionality
        - bool bias: Whether to include bias
        - float dropout: Dropout probability
        - bool bi_d: Bidirectional Input
        - bool batch_first: Dimension ordering of batched (i.e. (batches x seq. length x features) as opposed to (seq. length x batches x features))
        """
        super(base_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.num_layers = num_lstm_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.bi_d = bi_d
        self.bias = bias
        self.batch_first = batch_first
        self.ens_model = ens_model

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=bi_d,
            bias=bias,
            batch_first=batch_first,
        )

        self.linear = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x, hidden=None):
        x, _ = self.lstm(x, hidden)
        x = self.linear(x)
        return x
