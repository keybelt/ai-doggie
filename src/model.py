import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import sys
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = init_layer(nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=0))
        self.conv2 = init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
        self.conv3 = init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0))
        self.conv4 = init_layer(nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0))

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()

        self.shared_fc = init_layer(nn.Linear(self.flat_size, 512))

        self.actor_net = nn.Sequential(
            init_layer(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 2), std=0.01)
        )

        self.critic_net = nn.Sequential(
            init_layer(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 1), std=1.0)
        )

    def forward_features(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)

        x = F.relu(self.shared_fc(x), inplace=True)

        return self.actor_net(x), self.critic_net(x)

    def get_action(self, x):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    def __init__(self, model, lr, gamma, eps_clip, batch_size, ent_coef, epoch):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epoch
        self.ent_coef = ent_coef

    def update(self, states, actions, log_probs, rewards, dones, values, last_value, batch_is_human):
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

        full_actions = actions.to(device)
        full_log_probs = log_probs.to(device)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        total_policy_loss = 0
        total_entropy = 0
        update_count = 0

        n_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        total_steps = self.epochs * n_batches
        step_curr = 0
        print("")

        for epoch_i in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx].to(device).float().div(255.0).permute(0, 3, 1, 2).contiguous()

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
                ppo_loss = -torch.min(surr1, surr2)
                bc_loss = F.cross_entropy(logits, batch_actions, reduction='none')
                mask = batch_is_human[batch_idx].to(device).float()
                bc_term = (bc_loss * mask).mean()
                total_loss = ((ppo_loss * (1 - mask)).mean()) + (bc_term)
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(new_values.squeeze(), batch_returns)

                loss = total_loss + 0.5 * critic_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

                step_curr += 1
                percent = int((step_curr / total_steps) * 100)
                bar_len = 25
                filled = int(bar_len * step_curr // total_steps)
                bar = '█' * filled + '-' * (bar_len - filled)

                sys.stdout.write(f"\r    [Backprop] |{bar}| {percent}% (Epoch {epoch_i + 1}/{self.epochs})")
                sys.stdout.flush()

        del full_actions, full_log_probs, returns, batch_states

        return total_policy_loss / update_count, total_entropy / update_count