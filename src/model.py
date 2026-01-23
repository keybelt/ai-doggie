import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1: Capture fine details (edges, spikes)
        # Stride 2 reduces 332x588 -> 166x294
        self.conv1 = nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(24)

        # Layer 2: Form shapes (blocks, slopes)
        # Stride 2 reduces -> 83x147
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(48)

        # Layer 3: Complex objects (portals, orbs)
        # Stride 2 reduces -> 42x74
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer 4: High level spatial reasoning
        # Stride 2 reduces -> 21x37
        # NOTE: Total stride is 16. A 22px gap is now ~1.3px in feature space.
        # This is the limit; do not stride again.
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 1x1 CONV BOTTLENECK (The Speed Trick)
        # Instead of flattening 64 channels, we squash them to 16.
        # This cuts the Linear layer parameters by 75% while keeping spatial info.
        self.bottleneck = nn.Conv2d(64, 16, kernel_size=1, stride=1)

        # Calculate flat size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()
            # Expected: 16 channels * 21 * 37 = ~12,432 (vs your old ~99,000)

        # Actor-Critic Heads
        # We keep 512 hidden units to ensure it has memory capacity
        # to "overfit" the extreme demon patterns.
        self.fc = init_layer(nn.Linear(self.flat_size, 512))
        self.actor = init_layer(nn.Linear(512, 2), std=0.01)
        self.critic = init_layer(nn.Linear(512, 1), std=1)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = F.relu(self.bottleneck(x), inplace=True)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        x = F.relu(self.fc(x), inplace=True)
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        logits, value = self(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr=2.0e-4, gamma=0.99, eps_clip=0.2, batch_size=128):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 5

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