import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random
import gymnasium as gym

from minigrid.envs.multiroom_sound import MultiRoomEnvSound
from src.models import MinigridPolicyNetClassif
from src.arguments import parser
from src.algos.she import compute_discriminator_loss

# python she_noRL.py --env MiniGrid-MultiRoomEnvSound-N2-S4-v0 --batch_size 32 --learning_rate 0.0003 --seed 42 --use_sound

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

env = gym.make(args.env)
obs, info = env.reset()

with open("replay_buffer.txt", "r") as f:
    data = [json.loads(line) for line in f]

class ReplayBufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
    def __len__(self):
        return len(self.buffer)
    def __getitem__(self, idx):
        item = self.buffer[idx]
        img = torch.tensor(item['partial_obs'], dtype=torch.float32)
        audio = torch.tensor(item['sound'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 3:
            img = img.mean(dim=0, keepdim=True)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        if label.numel() > 1:
            label = label.view(-1)[0]
        else:
            label = label.squeeze()
        return img, audio, label

dataset = ReplayBufferDataset(data)
train_len = int(len(dataset) * 0.8)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(args.seed))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

observation_dict = {
    "image": torch.zeros(env.observation_space['image'].shape),
    "sound": torch.zeros(env.observation_space['sound'].shape)
}
num_actions = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
model = MinigridPolicyNetClassif(observation_dict, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

for epoch in range(args.unroll_length):
    model.train()
    train_loss, train_acc, train_total = 0, 0, 0
    train_intrinsic = 0
    for images, audios, _ in train_loader:
        images, audios = images.to(device), audios.to(device)
        optimizer.zero_grad()
        #discrim_loss, targets, preds = compute_discriminator_loss(model, images, None, audios)
        discrim_loss, targets, preds = compute_discriminator_loss(model, images, None, audios)
        if preds.dim() == 2 and preds.shape[1] != 1:
            preds = preds[:, 0]
        if preds.dim() > 1:
            preds = preds.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        loss = F.binary_cross_entropy(preds, targets)
        #loss = discrim_loss.mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds_bin = (preds > 0.5).float()
        train_acc += (preds_bin == targets).sum().item()
        train_total += targets.numel()
        train_intrinsic += targets.mean().item()
    train_loss /= len(train_loader)
    train_acc = train_acc / train_total if train_total > 0 else 0
    train_intrinsic /= len(train_loader)

    # Validation
    model.eval()
    test_loss, test_acc, test_total = 0, 0, 0
    test_intrinsic = 0
    with torch.no_grad():
        for images, audios, _ in test_loader:
            images, audios = images.to(device), audios.to(device)
            #discrim_loss, targets, preds = compute_discriminator_loss(model, images, None, audios)
            discrim_loss, targets, preds = compute_discriminator_loss(model, images, None, audios)
            if preds.dim() == 2 and preds.shape[1] != 1:
                preds = preds[:, 0]
            if preds.dim() > 1:
                preds = preds.view(-1)
            if targets.dim() > 1:
                targets = targets.view(-1)
            loss = F.binary_cross_entropy(preds, targets)
            #loss = discrim_loss.mean()
            test_loss += loss.item()
            preds_bin = (preds > 0.5).float()
            test_acc += (preds_bin == targets).sum().item()
            test_total += targets.numel()
            test_intrinsic += targets.mean().item()
    test_loss /= len(test_loader)
    test_acc = test_acc / test_total if test_total > 0 else 0
    test_intrinsic /= len(test_loader)

    print(f"Epoch {epoch+1}/{args.unroll_length} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | "
          f"Train Intrinsic: {train_intrinsic:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f} | "
          f"Test Intrinsic: {test_intrinsic:.4f}")

print("Entraînement terminé.")