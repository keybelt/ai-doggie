import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from model import GDBehavioralCloningModel

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 4
SEQ_LEN = 60
LEARNING_RATE = 3e-4
ACTION_DROPOUT_RATE = 0.5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GDSequenceIterableDataset(IterableDataset):
    def __init__(self, file_paths, seq_len=60, batch_size=4):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        import random
        random.shuffle(self.file_paths)
        file_queue = list(self.file_paths)
        active_streams = [None] * self.batch_size

        while True:
            batch_frames, batch_actions, batch_resets = [], [], []
            active_count = 0

            for i in range(self.batch_size):
                if active_streams[i] is None and file_queue:
                    active_streams[i] = self._stream_file(file_queue.pop(0))

                stream = active_streams[i]

                if stream is not None:
                    try:
                        frames, actions, is_first = next(stream)
                        batch_frames.append(frames)
                        batch_actions.append(actions)
                        batch_resets.append(is_first)
                        active_count += 1
                    except StopIteration:
                        if file_queue:
                            active_streams[i] = self._stream_file(file_queue.pop(0))
                            frames, actions, is_first = next(active_streams[i])
                            batch_frames.append(frames)
                            batch_actions.append(actions)
                            batch_resets.append(is_first)
                            active_count += 1
                        else:
                            active_streams[i] = None

            if active_count == 0:
                break

            if active_count < self.batch_size:
                break

            yield np.stack(batch_frames), np.stack(batch_actions), np.array(batch_resets)

    def _stream_file(self, path):
        try:
            with np.load(path) as data:
                frames, actions = data['frames'], data['actions']
                num_chunks = (len(frames) - 1) // self.seq_len

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * self.seq_len
                    chunk_f = frames[start : start + self.seq_len]
                    chunk_a = actions[start : start + self.seq_len + 1]
                    yield np.transpose(chunk_f, (0, 3, 1, 2)), chunk_a, (chunk_idx == 0)
        except Exception as e:
            print(f"[Skip] Error loading {path}: {e}")


def train():
    file_paths = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not file_paths:
        print(f"❌ No .npz files found in {DATASET_DIR}. Run record.py first!")
        sys.exit(1)

    print(f"📂 Found {len(file_paths)} golden runs. Initializing Dataset...")

    dataset = GDSequenceIterableDataset(file_paths, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    print("🧠 Initializing Impala ResNet + GRU Model...")
    model = GDBehavioralCloningModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"🚀 Training Started on {DEVICE.type.upper()} Backend")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        hidden_state = None

        for batch_idx, (b_frames, b_actions, b_is_first) in enumerate(dataloader):

            if hidden_state is not None:
                b_is_first_tensor = torch.from_numpy(b_is_first).to(DEVICE)
                keep_mask = (~b_is_first_tensor).to(dtype=torch.float32)

                keep_mask = keep_mask.unsqueeze(0).unsqueeze(-1)
                hidden_state = hidden_state * keep_mask

            b_frames = b_frames.to(DEVICE, dtype=torch.float32).div(255.0)
            prev_actions = b_actions[:, :-1].to(DEVICE, dtype=torch.long)
            target_actions = b_actions[:, 1:].to(DEVICE, dtype=torch.long)

            drop_mask = torch.rand(prev_actions.shape, device=DEVICE) < ACTION_DROPOUT_RATE
            prev_actions = prev_actions.masked_fill(drop_mask, 2)

            optimizer.zero_grad(set_to_none=True)

            logits, hidden_state = model(b_frames, prev_actions, hidden_state)

            if hidden_state is not None:
                hidden_state = hidden_state.detach()

            valid_logits = logits.reshape(-1, 2)
            valid_actions = target_actions.reshape(-1)

            loss = criterion(valid_logits, valid_actions)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch}/{EPOCHS}] | Batch {batch_idx} | Loss: {loss.item():.4f}")
                sys.stdout.flush()

        avg_loss = epoch_loss / max(1, batch_count)
        print(f"\n✅ Epoch {epoch} Completed | Average Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"gd_model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "gd_model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training Complete! Final weights saved to {final_path}")


if __name__ == "__main__":
    train()