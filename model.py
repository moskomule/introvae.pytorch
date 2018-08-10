from torch import nn
from torch.nn import functional as F


class MiniEncoder(nn.Module):
    def __init__(self):
        super(MiniEncoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)


class MiniDecoder(nn.Module):
    def __init__(self):
        super(MiniDecoder, self).__init__()

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))


MODELS = {"mini": (MiniEncoder, MiniDecoder)}
