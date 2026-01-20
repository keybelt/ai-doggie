import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # [STEM] 16 Channels (Standard)
        self.conv1 = nn.Conv2d(12, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # [BODY] "Hybrid" Configuration
        # Progression: 16 -> 24 -> 32 -> 64
        # We cap at 64. This is the sweet spot for 120Hz on M4 Pro.

        # Layer 1: 16 -> 24 (Stride 2)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)

        # Layer 2: 24 -> 32 (Stride 2)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        # Layer 3: 32 -> 64 (Stride 1 - High Res Logic)
        # We use Stride 1 here immediately to get a higher resolution semantic map earlier.
        # This results in a grid of 42x74 (High Res) or 21x37 (Low Res).
        # Let's stick to the verified 21x37 grid by using Stride 2 once more.

        # Correction: Layer 3 needs to be Stride 2 to hit 21x37.
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # Layer 4: 64 -> 64 (Stride 1 - Processing)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        # [PROJECTION]
        # 64 -> 16 Channels.
        self.projection = nn.Conv2d(64, 16, kernel_size=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()
            # Dimensions: [1, 16, 21, 37] -> 12,432 features.

        # [HEAD]
        # 256 Neurons. Fast and sufficient.
        self.fc = nn.Linear(self.flat_size, 256)
        self.actor = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # Added the processing layer
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
    def __init__(self, model, lr=3e-4, gamma=0.995, eps_clip=0.2, batch_size=256):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 4

    def update(self, states, actions, log_probs, rewards, dones, values):
        # 1. Calculate Returns (CPU is fine for this size)
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        # Move metadata to GPU
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        full_actions = torch.stack(actions).to(device).squeeze()
        full_log_probs = torch.stack(log_probs).to(device).squeeze()

        # [OPTIMIZATION START] ------------------------
        # Convert List of Arrays -> Single Tensor (CPU, uint8) ONCE.
        # This takes ~1-2 seconds but saves ~40 seconds of loop overhead.
        # Memory: ~4.7 GB (Safe for M4 Pro)
        full_states = torch.from_numpy(np.stack(states))
        # [OPTIMIZATION END] --------------------------

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

                # [FAST SLICING]
                # 1. Slice the pre-allocated CPU tensor (Instant, View only)
                batch_states = full_states[batch_idx]

                # 2. Move 256 frames to GPU -> Float -> Permute -> Contiguous
                # This pipeline is optimal for MPS bandwidth.
                batch_states = batch_states.to(device).float().div(255.0)
                batch_states = batch_states.permute(0, 3, 1, 2).contiguous()

                batch_actions = full_actions[batch_idx]
                batch_old_log_probs = full_log_probs[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward / Backward
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
                loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # Cleanup immediate garbage
                del batch_states

                # Tiny sleep for capture thread
                time.sleep(0.001)

                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        # [CLEANUP]
        # Delete the massive 4.7GB tensor immediately to free RAM for gameplay
        del full_states
        import gc
        gc.collect()  # Force Python to kill the objects
        torch.mps.empty_cache()  # Force MPS to release VRAM

        return total_policy_loss / update_count, total_entropy / update_count