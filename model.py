import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, complexity):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, complexity)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(complexity, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
