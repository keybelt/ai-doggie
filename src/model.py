import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. The Eyes (CNN) ---
        self.conv1 = nn.Conv2d(12, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # --- 2. The Compressor ---
        # We keep this for efficiency
        self.compressor = nn.AdaptiveMaxPool2d((12, 20))

        # --- 3. Dynamic Size Calc ---
        with torch.no_grad():
            # [CRITICAL CHANGE]
            # We now expect the input to be half-size (166x294) due to slicing.
            # Original: 332x588 -> Sliced: 166x294
            dummy = torch.zeros(1, 12, 166, 294)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.compressor(x)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, 256)
        self.actor = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.compressor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        # [CRITICAL] Inference needs to slice too!
        # If x is full res (332x588), we slice it to match training (166x294)
        if x.shape[-1] == 588:
            x = x[:, :, ::2, ::2]

        logits, value = self(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr=2.5e-4, gamma=0.99, eps_clip=0.2, batch_size=1024):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 4

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

        # Metrics tracking for the dashboard
        total_policy_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                raw_chunk = full_states[batch_idx]
                sliced_chunk = raw_chunk[:, :, ::2, ::2]

                batch_states = sliced_chunk.to(device).float().div(255.0)
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

                # Total Loss (0.01 is the entropy coefficient / 'curiosity weight')
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record metrics
                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        # --- THE FIX: Return metrics to train.py ---
        avg_loss = total_policy_loss / update_count if update_count > 0 else 0
        avg_entropy = total_entropy / update_count if update_count > 0 else 0

        return avg_loss, avg_entropy