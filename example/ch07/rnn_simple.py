import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, x_dim: int, h_size: int):
        super(SimpleRNN, self).__init__()
        self.W_x = nn.Linear(x_dim, h_size, dtype=float)
        self.W_h = nn.Linear(h_size, h_size, dtype=float)
        self.input_dim = x_dim
        self.hidden_size = h_size

        self.fc1 = nn.Linear(h_size, 40, dtype=float)
        self.fc2 = nn.Linear(40, 10, dtype=float)
        self.out_layer = nn.Linear(10, 2, dtype=float)

    # def __call__(self, x):
    #     return self.forward(x)

    def step(self, x, h):
        # h^{t} = (U * h^{t-1} + W * x + b1 + b2)
        wx = self.W_x(x)
        ht = self.W_h(h) + wx
        out = torch.tanh(ht)
        return out, ht

    def forward(self, x):
        # x (time_steps, inputs(1, input_size))
        # h_t (1, hidden_size)
        # output (input_size, hidden_size)
        seq_len = x.shape[0]

        ht = self.W_x(x[0])
        output = torch.empty(self.input_dim, self.hidden_size)

        for i in range(1, seq_len):
            output, ht = self.step(x[i, :], ht)

        f1 = torch.relu(self.fc1(output))
        f2 = torch.relu(self.fc2(f1))
        r_out = self.out_layer(f2)
        return r_out
