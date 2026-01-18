import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flat_size = x.numel()

        self.fc = nn.Linear(self.flat_size, 512)

        self.actor = nn.Linear(512, 2)
        self.critic = nn.Linear(512, 1)

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
    def __init__(self, model, lr=2.5e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()

    def update(self, states, actions, log_probs, rewards, dones, values):
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.stack(states).to(device).detach()
        old_actions = torch.stack(actions).to(device).detach()
        old_log_probs = torch.stack(log_probs).to(device).detach()

        for _ in range(4):
            logits, new_values = self.model(old_states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            advantage = returns - new_values.squeeze().detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(new_values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()