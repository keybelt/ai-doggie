import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DSResBlock(nn.Module):
    """
    Depthwise Separable Residual Block.
    Maintains the 'Smarts' (Depth/Non-linearities) of a standard ResBlock
    but uses ~85% fewer parameters and FLOPs.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # --- Layer 1 ---
        # 1a. Depthwise (Spatial features only)
        self.dw1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                             padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # 1b. Pointwise (Mix channels)
        self.pw1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # --- Layer 2 ---
        # 2a. Depthwise
        self.dw2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                             padding=1, groups=out_ch, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        # 2b. Pointwise
        self.pw2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x

        # Layer 1
        x = F.relu(self.bn1(self.dw1(x)))
        x = F.relu(self.bn2(self.pw1(x)))

        # Layer 2
        x = F.relu(self.bn3(self.dw2(x)))
        x = self.bn4(self.pw2(x))

        # Residual add
        x += self.shortcut(residual)
        return F.relu(x)


class GDPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # [STEM] Reduced channels from 32 -> 24 for speed/memory
        # Input: 12 channels (4 frames x 3 colors)
        self.conv1 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)

        # [IMPALA-LITE STACK]
        # We use a slimmer channel progression: 24 -> 32 -> 48 -> 64.
        # This drastically reduces VRAM usage (solving the yellow/red pressure)
        # while keeping the network deep (4 layers of logic).

        self.layer1 = DSResBlock(24, 32, stride=2)  # 83x147
        self.layer2 = DSResBlock(32, 48, stride=2)  # 42x74
        self.layer3 = DSResBlock(48, 64, stride=2)  # 21x37 (Grid established)
        self.layer4 = DSResBlock(64, 64, stride=1)  # 21x37 (Maintain resolution)

        # [PROJECTION HEAD]
        # 64 Channels -> 16 Channels.
        # We Keep the 21x37 grid so the bot knows EXACTLY where spikes are.
        # 16 channels allows encoding of: [Spike, Block, Portal, Speed, Mini, Dual, etc.]
        self.projection = nn.Conv2d(64, 16, kernel_size=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 332, 588)
            x = self.forward_features(dummy)
            self.flat_size = x.numel()
            # Dimensions: [1, 16, 21, 37] -> 12,432 features.
            # This is lightweight and will run extremely fast on M4 Pro.

        self.fc = nn.Linear(self.flat_size, 512)
        self.actor = nn.Linear(512, 2)
        self.critic = nn.Linear(512, 1)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.projection(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)

        # [CRITICAL] Use flatten(1) to handle MPS non-contiguous memory
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
    def __init__(self, model, lr=5e-4, gamma=0.995, eps_clip=0.2, batch_size=256):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = 3

    def update(self, states, actions, log_probs, rewards, dones, values):
        # 1. Calculate Returns (Can be done on CPU or GPU, lightweight)
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        # Tensors for metadata are small, so we can move them all at once
        full_actions = torch.stack(actions).to(device).squeeze()
        full_log_probs = torch.stack(log_probs).to(device).squeeze()

        # Returns handling
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        dataset_size = len(states)
        indices = np.arange(dataset_size)  # Use numpy for indices

        total_policy_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)  # Shuffle indices

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # [CRITICAL MEMORY FIX]
                # 1. Slice the list of Numpy Arrays (CPU RAM)
                batch_states_np = [states[i] for i in batch_idx]

                # 2. Stack into one big Numpy Array (CPU RAM)
                # 3. Convert to Tensor -> Move to GPU -> Float -> Normalize (VRAM)
                # This pipeline ensures we never hold 20GB of floats in memory.
                batch_states = torch.from_numpy(np.stack(batch_states_np)).to(device).float().div(255.0)

                # Permute to (Batch, Channels, Height, Width) for the CNN
                batch_states = batch_states.permute(0, 3, 1, 2).contiguous()

                batch_actions = full_actions[batch_idx]
                batch_old_log_probs = full_log_probs[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
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

                # Cleanup to keep VRAM clean
                del batch_states

                # Tiny sleep to let Capture Engine run
                time.sleep(0.001)

                total_policy_loss += actor_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        # Final Flush.flatten(1)
        torch.mps.empty_cache()
        return total_policy_loss / update_count, total_entropy / update_count