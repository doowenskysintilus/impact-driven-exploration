import json
import torch
import numpy as np
import torch
import gymnasium as gym
#from src.models import MinigridPolicyNetClassif as models
#from src.models import MinigridPolicyNetClassif
#MinigridPolicyNetClassif = models.MinigridPolicyNetClassif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import gymnasium as gym
from src.models import MinigridPolicyNetClassif
from minigrid.envs.multiroom_sound import MultiRoomEnvSound
import matplotlib.pyplot as plt
import torch.nn.functional as F



env = gym.make("MiniGrid-MultiRoomEnvSound-N2-S4-v0")
obs, info = env.reset()

model = MinigridPolicyNetClassif(env.observation_space, env.action_space.n)
model.eval()


replay_buffer = []
with open("replay_buffer.txt", "r") as f:
    for line in f:
        replay_buffer.append(json.loads(line))

sounds = torch.tensor([t['sound'] for t in replay_buffer], dtype=torch.float32)
images = torch.tensor([t['partial_obs'] for t in replay_buffer], dtype=torch.float32)
#labels = torch.tensor([t['label'] for t in replay_buffer], dtype=torch.float32).unsqueeze(1)
labels = torch.tensor([t['label'] for t in replay_buffer], dtype=torch.float32)

inputs = {
    'partial_obs': images.to(device),
    'sound': sounds.to(device),
}
outputs, _ = model(inputs)
preds = outputs['binary_class_output']
if preds.shape != labels.shape:
    labels = labels.view(preds.shape)

losses = F.binary_cross_entropy(preds.cpu(), labels.cpu(), reduction='none')


predicted_labels = (preds > 0.5).float()
accuracy = (predicted_labels.cpu() == labels.cpu()).float().mean().item()

import numpy as np

window = 100  # taille de la fenêtre pour la moyenne glissante
losses_np = losses.detach().cpu().numpy().flatten()
moving_avg = np.convolve(losses_np, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,4))
plt.plot(losses_np, alpha=0.3, label="Loss brute")
plt.plot(np.arange(window-1, len(losses_np)), moving_avg, color='red', label=f"Loss moyenne glissante (window={window})")
plt.xlabel("Index dans le replay buffer")
plt.ylabel("Loss")
plt.title("Loss du discriminateur sur le replay buffer")
plt.legend()
plt.show()

accuracies = (predicted_labels.cpu() == labels.cpu()).float().numpy().flatten()

moving_acc = np.convolve(accuracies, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,4))
plt.plot(accuracies, alpha=0.3, label="Accuracy brute")
plt.plot(np.arange(window-1, len(accuracies)), moving_acc, color='green', label=f"Accuracy moyenne glissante (window={window})")
plt.xlabel("Index dans le replay buffer")
plt.ylabel("Accuracy")
plt.title("Accuracy du discriminateur sur le replay buffer")
plt.legend()
plt.show()

print("Replay buffer accuracy:", accuracy)
print("Loss moyenne :", losses.mean().item())