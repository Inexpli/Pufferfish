import torch
import torch.nn as nn
import torch.nn.functional as F

HISTORY_LENGTH = 4
GLOBAL_CHANNELS = 1
INPUT_CHANNELS = 17 * HISTORY_LENGTH + GLOBAL_CHANNELS
RESIDUAL_FILTERS = 128
NUM_RESIDUAL_BLOCKS = 8
NUM_MOVES = 4672

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(INPUT_CHANNELS, RESIDUAL_FILTERS, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(RESIDUAL_FILTERS)

        self.res_blocks = nn.Sequential(*[ResidualBlock(RESIDUAL_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)])

        self.conv_policy = nn.Conv2d(RESIDUAL_FILTERS, 73, kernel_size=1)
        self.fc_policy = nn.Linear(8*8*73, NUM_MOVES)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        policy = F.relu(self.conv_policy(x))
        policy = policy.view(policy.size(0), -1)
        return self.fc_policy(policy)