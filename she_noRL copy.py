import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random
import gymnasium as gym

from minigrid.envs.multiroom_sound import MultiRoomEnvSound
from src.models import MinigridPolicyNetClassif
from src.arguments import parser 

#python she_noRL.py --env MiniGrid-MultiRoomEnvSound-N2-S4-v0 --batch_size 32 --learning_rate 0.0003 --seed 42 --use_sound

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
criterion = nn.BCELoss()

#Entraînement
for epoch in range(args.unroll_length):
    model.train()
    train_loss, train_acc = 0, 0
    for images, audios, labels in train_loader:
        images, audios, labels = images.to(device), audios.to(device), labels.to(device)
        optimizer.zero_grad()
        #outputs = model(images, audios)
        outputs = model({'image': images, 'sound': audios})
        #print(type(outputs))
        while isinstance(outputs, (dict, tuple)):
            if isinstance(outputs, dict):
                outputs = outputs['binary_class_output']
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
        #print(type(outputs))
        # Correction de la forme
        if outputs.dim() == 2:
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            elif outputs.shape[1] == outputs.shape[0]:
                outputs = torch.diagonal(outputs, 0)
            else:
                # On prend la première colonne si c'est [batch_size, N]
                outputs = outputs[:, 0]
        elif outputs.dim() > 1:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        if outputs.shape != labels.shape:
            raise ValueError(f"Output shape {outputs.shape} and label shape {labels.shape} do not match")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = (outputs > 0.5).float()
        train_acc += (preds == labels).sum().item()
    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)

    #Validation
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for images, audios, labels in test_loader:
            images, audios, labels = images.to(device), audios.to(device), labels.to(device)
            #outputs = model(images, audios)
            outputs = model({'image': images, 'sound': audios})
            while isinstance(outputs, (dict, tuple)):
                if isinstance(outputs, dict):
                    outputs = outputs['binary_class_output']
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
            #print(type(outputs))
            if outputs.dim() == 2:
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.shape[1] == outputs.shape[0]:
                    outputs = torch.diagonal(outputs, 0)
                else:
                    # On prend la première colonne si c'est [batch_size, N]
                    outputs = outputs[:, 0]
            elif outputs.dim() > 1:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            if outputs.shape != labels.shape:
                raise ValueError(f"Output shape {outputs.shape} and label shape {labels.shape} do not match")
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (outputs > 0.5).float()
            test_acc += (preds == labels).sum().item()
    test_loss /= len(test_loader)
    test_acc /= len(test_dataset)

    print(f"Epoch {epoch+1}/{args.unroll_length} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}")

print("Entraînement terminé.")