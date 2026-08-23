import torch.nn as nn


class ClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=784, out_features=256)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=256, out_features=128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
