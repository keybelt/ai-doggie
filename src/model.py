import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GeometryDashPolicy(nn.Module):
    def __init__(self, action_dim=2):
        super(GeometryDashPolicy, self).__init__()

        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_input_dim = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.fc_input_dim, 512)

        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 12, 332, 588)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)

        return int(np.prod(x.size()))

    def forward(self, x):
        x = x / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value


device = torch.device("mps")
model = GeometryDashPolicy().to(device)

