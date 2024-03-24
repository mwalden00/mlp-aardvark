import torch
import torch.nn as nn


class CNN_Net(nn.Module):
    def __init__(self, input_shape, kernel_size):
        super(CNN_Net, self).__init__()

        x = torch.zeros(input_shape)
        out = x
        self.conv1d = nn.Conv1d(out.shape[0], 1, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1*out.shape[1], 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
