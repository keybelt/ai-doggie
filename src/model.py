import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x + res


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return F.relu(x)


class GDBehavioralCloningModel(nn.Module):
    def __init__(self, num_actions=2, action_vocab_size=3, hidden_size=256, action_emb_dim=16):
        super().__init__()

        self.hidden_size = hidden_size

        self.block1 = ImpalaBlock(3, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 64)
        self.block4 = ImpalaBlock(64, 64)
        self.block5 = ImpalaBlock(64, 64)

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 332, 588)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, hidden_size)

        self.action_emb = nn.Embedding(action_vocab_size, action_emb_dim)

        self.gru = nn.GRU(hidden_size + action_emb_dim, hidden_size, num_layers=1, batch_first=True)

        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x, prev_actions, hidden_state=None):
        batch_size, seq_len, c, h, w = x.size()

        x = x.reshape(batch_size * seq_len, c, h, w)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.reshape(batch_size * seq_len, -1)
        x = F.relu(self.fc(x), inplace=False)

        cnn_features = x.reshape(batch_size, seq_len, self.hidden_size)

        action_features = self.action_emb(prev_actions)

        gru_input = torch.cat([cnn_features, action_features], dim=-1)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        out, next_hidden = self.gru(gru_input, hidden_state)

        logits = self.policy_head(out)

        return logits, next_hidden