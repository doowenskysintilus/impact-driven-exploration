import torch
from types import SimpleNamespace
from gym.spaces import Dict, Box
import numpy as np
from src.algos.she import learn
from src.algos.she import train  


# Mock des flags
flags = SimpleNamespace()
flags.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flags.use_sound = True
flags.use_fullobs_policy = False
flags.use_fullobs_intrinsic = False
flags.unroll_length = 5
flags.batch_size = 2
flags.intrinsic_reward_coef = 0.1
flags.forward_loss_coef = 0.2
flags.inverse_loss_coef = 0.2
flags.baseline_cost = 0.5
flags.entropy_cost = 0.001
flags.no_reward = False
flags.discounting = 0.99
flags.max_grad_norm = 40


obs_space = Dict({
    "image": Box(low=0, high=1.0, shape=(7, 7, 3), dtype=np.uint8), 
    "sound": Box(low=-1.0, high=1.0, shape=(32,), dtype=np.float32)
})

"""obs_space = Dict({
    "image": torch.zeros(7, 7, 3),   #(H, W, C) ->image 7x7x3
    "sound": torch.zeros(10)         #vecteur son de dimension 10
})"""

class DummyActionSpace:
    def __init__(self, n):
        self.n = n

action_space = DummyActionSpace(n=7)

from src.models_sound import MinigridPolicyNet_Sound, MinigridStateEmbeddingNet, MinigridForwardDynamicsNet, MinigridInverseDynamicsNet

model = MinigridPolicyNet_Sound(obs_space, action_space.n).to(flags.device)
state_embedding_model = MinigridStateEmbeddingNet(obs_space).to(flags.device)
forward_dynamics_model = MinigridForwardDynamicsNet(action_space.n).to(flags.device)
inverse_dynamics_model = MinigridInverseDynamicsNet(action_space.n).to(flags.device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
state_embedding_optimizer = torch.optim.Adam(state_embedding_model.parameters(), lr=0.001)
forward_dynamics_optimizer = torch.optim.Adam(forward_dynamics_model.parameters(), lr=0.001)
inverse_dynamics_optimizer = torch.optim.Adam(inverse_dynamics_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# Dummy batch (fake data with correct shapes)
batch = {
    'partial_obs': {
        'image': torch.randint(0, 256, (flags.unroll_length + 1, flags.batch_size, 7, 7, 3), dtype=torch.uint8),
        'sound': torch.randn(flags.unroll_length + 1, flags.batch_size, 32),
    },
    'action': torch.randint(0, action_space.n, (flags.unroll_length + 1, flags.batch_size)),
    'reward': torch.randn(flags.unroll_length, flags.batch_size),
    'done': torch.randint(0, 2, (flags.unroll_length, flags.batch_size), dtype=torch.bool),
    'policy_logits': torch.randn(flags.unroll_length + 1, flags.batch_size, action_space.n),
    'episode_return': torch.randn(flags.unroll_length, flags.batch_size),
    'episode_state_count': torch.ones(flags.unroll_length + 1, flags.batch_size),
}

#Récupération des données du batch pour les inputs
inputs = {
    'image': batch['partial_obs']['image'],  # Image
    'sound': batch['partial_obs']['sound'],  # Son
}

state_emb = state_embedding_model(inputs)


# Dummy initial state
initial_agent_state = model.initial_state(batch_size=flags.batch_size)

print(f"Batch size: {batch['partial_obs']['image'].size(1)}")
print(f"Batch size: {batch['partial_obs']['sound'].size(1)}")
print(f"Batch size: {batch['action'].size(1)}")


# Test learn
stats = learn(
    actor_model=model,
    model=model,
    state_embedding_model=state_embedding_model,
    forward_dynamics_model=forward_dynamics_model,
    inverse_dynamics_model=inverse_dynamics_model,
    batch=batch,
    initial_agent_state=initial_agent_state,
    optimizer=optimizer,
    state_embedding_optimizer=state_embedding_optimizer,
    forward_dynamics_optimizer=forward_dynamics_optimizer,
    inverse_dynamics_optimizer=inverse_dynamics_optimizer,
    scheduler=scheduler,
    flags=flags
)

print("Test terminé avec succès. Stats:")
print(stats)
