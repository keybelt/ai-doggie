import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(12, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # Dynamic Flat Size Calc
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, 256)
        self.actor = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
    def __init__(self, model, lr=2.5e-4, gamma=0.99, eps_clip=0.2, epochs=4, batch_size=256):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size  # Process small chunks to save RAM
        self.mse_loss = nn.MSELoss()

    def update(self, states, actions, log_probs, rewards, dones, values):
        # 1. Calculate Returns (Monte Carlo)
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        # Convert simple lists to tensors (Cheap)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Note: 'states' is still a list of CPU tensors here. We DON'T stack it yet.
        # This keeps it at ~9GB (uint8) instead of 35GB (float32).

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        # 2. PPO Epochs with Mini-Batches
        for _ in range(self.epochs):
            np.random.shuffle(indices)

            # Iterate in chunks (Mini-Batches)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # --- MEMORY SAFE LOADING ---
                # Only convert THIS small chunk to Float/GPU
                batch_states = torch.stack([states[i] for i in batch_idx]).to(device).float().div(255.0).squeeze(1)
                batch_actions = torch.stack([actions[i] for i in batch_idx]).to(device).squeeze(1)
                batch_old_log_probs = torch.stack([log_probs[i] for i in batch_idx]).to(device).squeeze(1)
                batch_returns = returns[batch_idx]

                # Forward Pass
                logits, new_values = self.model(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratios
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Loss
                advantage = batch_returns - new_values.squeeze()
                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(new_values.squeeze(), batch_returns)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()