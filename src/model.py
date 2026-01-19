import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Standard ResBlock but without heavy padding overhead
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # --- THE STABILITY STEM ---
        # 12 -> 48 Filters: Fast initial reduction
        self.conv1 = nn.Conv2d(12, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        # --- THE WIDE BODY ---
        # Layer 1: 48 filters
        self.layer1 = ResBlock(48)

        # Layer 2: 96 filters
        self.conv_up1 = nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2 = ResBlock(96)

        # Layer 3: 192 filters (The "Demon Logic" sweet spot)
        # We cap at 192 to eliminate those 26 drops while keeping the 256-style IQ
        self.conv_up2 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3 = ResBlock(192)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, 512)
        self.actor = nn.Linear(512, 2)
        self.critic = nn.Linear(512, 1)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        x = F.relu(self.conv_up1(x))
        x = self.layer2(x)
        x = F.relu(self.conv_up2(x))
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        logits, value = self(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr=1e-4, gamma=0.995, eps_clip=0.1, batch_size=256):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 3

    def update(self, states, actions, log_probs, rewards, dones, values):
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        full_states = torch.stack(states)
        full_actions = torch.stack(actions).to(device).squeeze()
        full_log_probs = torch.stack(log_probs).to(device).squeeze()

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

                raw_chunk = full_states[batch_idx]
                batch_states = raw_chunk.to(device).float().div(255.0)
                batch_actions = full_actions[batch_idx]
                batch_old_log_probs = full_log_probs[batch_idx]
                batch_returns = returns[batch_idx]

                logits, new_values = self.model(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                advantage = batch_returns - new_values.squeeze()
                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(new_values.squeeze(), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        return total_policy_loss / update_count, total_entropy / update_count