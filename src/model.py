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
        self.conv1 = nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()

        self.fc1 = init_layer(nn.Linear(self.flat_size, 1024))
        self.fc2 = init_layer(nn.Linear(1024, 512))
        self.actor = init_layer(nn.Linear(512, 2), std=0.01)
        self.critic = init_layer(nn.Linear(512, 1), std=1)

    def forward_features(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return self.actor(x), self.critic(x)

    def get_action(self, x):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.995, eps_clip=0.15, batch_size=512):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 4

    def update(self, states, actions, log_probs, rewards, dones, values, last_value):
        torch.mps.empty_cache()

        gae_lambda = 0.95

        rewards = rewards.to(device)
        dones = dones.to(device)
        values = values.to(device)

        advantages = []
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + self.gamma * gae_lambda * next_non_terminal * last_advantage
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

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
                batch_advantages = advantages[batch_idx]

                logits, new_values = self.model(batch_states)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

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

        return total_policy_loss / update_count, total_entropy / update_count