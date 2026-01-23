import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.sa(out)
        out += residual
        return F.relu(out)


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock(64)

        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.res3a = ResidualBlock(96)
        self.res3b = ResidualBlock(96)

        self.projection = nn.Conv2d(96, 16, kernel_size=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, 512)
        self.actor = nn.Linear(512, 2)
        self.critic = nn.Linear(512, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3a(x)
        x = self.res3b(x)
        x = F.relu(self.projection(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        logits, value = self(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr=2.5e-4, gamma=0.995, eps_clip=0.2, batch_size=128):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 3

    def update(self, states, actions, log_probs, rewards, dones, values):
        torch.mps.empty_cache()

        returns = []
        discounted_sum = 0
        rewards_list = rewards.tolist()
        dones_list = dones.tolist()

        for reward, is_done in zip(reversed(rewards_list), reversed(dones_list)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        full_states = states
        full_actions = actions.to(device)
        full_log_probs = log_probs.to(device)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        total_policy_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = full_states[batch_idx].to(device, non_blocking=True).float().div(255.0)
                batch_states = batch_states.permute(0, 3, 1, 2).contiguous()

                batch_actions = full_actions[batch_idx]
                batch_old_log_probs = full_log_probs[batch_idx]
                batch_returns = returns[batch_idx]

                logits, new_values = self.model(batch_states)

                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                advantage = (batch_returns - new_values.squeeze()).detach()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(new_values.squeeze(), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        del full_actions, full_log_probs, returns, batch_states
        import gc
        gc.collect()
        torch.mps.empty_cache()

        return total_policy_loss / update_count, total_entropy / update_count