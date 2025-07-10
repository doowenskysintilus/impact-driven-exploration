# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np 


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FullObsMinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(FullObsMinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim)
                
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        ##Because Fully_observed
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )
        

        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple()
  
    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def create_embeddings(self, x, id):
        #indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        #object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() #select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location 

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        agent_loc = self.agent_loc(inputs["frame"])
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]

        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)
        
        core_output = core_input
        core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

class FullObsMinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(FullObsMinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) 
        
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        ##Because Fully_observed
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )
        
        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        
    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def create_embeddings(self, x, id):
        #indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        #object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() #select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location 

    def forward(self, inputs, next_state=False):
        # -- [unroll_length x batch_size x height x width x channels]
        if next_state:
            x = inputs["frame"][1:]
        else:
            x = inputs["frame"][:-1]
        T, B, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        if next_state:
            agent_loc = self.agent_loc(inputs["frame"][1:])
            carried_col = inputs["carried_col"][1:]
            carried_obj = inputs["carried_obj"][1:]
        else:
            agent_loc = self.agent_loc(inputs["frame"][:-1])
            carried_col = inputs["carried_col"][:-1]
            carried_obj = inputs["carried_obj"][:-1]

        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        return core_input

"""def normalize_input(x):
    return x.float() / 255.0

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)"""


class MinigridPolicyNetClassif(nn.Module):
    def __init__(self, observation_dict, num_actions):
        super(MinigridPolicyNetClassif, self).__init__()
        self.observation_shape = observation_dict["image"].shape
        self.observation_shape_sound = observation_dict["sound"].shape
        self.num_actions = num_actions

        init_ = lambda m: self.init(m)

        # Feature for image
        self.feat_extract = nn.Sequential(
            #LambdaLayer(normalize_input),
            nn.Identity(),
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ReLU(),
        )

        # Sound
        self.sound_mlp = nn.Sequential(
            init_(nn.Linear(self.observation_shape_sound[0], 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 8))
        )

        # Main fully connected layers
        self.fc = nn.Sequential(
            init_(nn.Linear(32*1*1 + 8, 1024)),
            #nn.LayerNorm(1024),
            nn.ReLU(),
            #nn.Dropout(0.3),
            init_(nn.Linear(1024, 1024)),
            #nn.LayerNorm(1024),
            nn.ReLU(),
            #nn.Dropout(0.3)
        )

        #Output
        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))

        # Binary classifier
        self.binary_classifier = nn.Sequential(
            init_(nn.Linear(1024, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            #nn.Dropout(0.3),
            init_(nn.Linear(128, 1)),
            nn.Sigmoid()
        )

    def init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        return m
    
    def initial_state(self, batch_size):
        return (torch.zeros(1, batch_size, 1),)
    
    """ forward(self, inputs, core_state=()):
        # Traitement de l'image
        if 'image' in inputs:
            x = inputs['image']
        elif 'partial_obs' in inputs:
            x = inputs['partial_obs']
        else:
            raise KeyError("inputs must contain 'image' or 'partial_obs'.")

        T, B, H, W, C = x.shape
        
        x = x.view(T * B, H, W, C).permute(0, 3, 1, 2)  # [T*B, C, H, W]
        x = x.float() #/ 255.0  #Normalization
        x = self.feat_extract(x)
        x = x.view(T * B, -1)

        #Traitement du son
        x_sound = inputs['sound'].view(T * B, -1).float()
        x_sound = self.sound_mlp(x_sound)

        #Fusion
        x = torch.cat([x, x_sound], dim=1)
        x = self.fc(x)

        #Sorties
        policy_logits = self.policy(x).view(T, B, -1)
        baseline = self.baseline(x).view(T, B)
        binary_output = self.binary_classifier(x).view(T, B)

        action = torch.multinomial(F.softmax(policy_logits.view(-1, self.num_actions), dim=1), 1) if self.training \
            else torch.argmax(policy_logits, dim=2)

        return dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action.view(T, B),
            binary_class_output=binary_output
        ), core_state"""
    def forward(self, inputs, core_state=()):
        # Traitement de l'image
        if 'image' in inputs:
            x = inputs['image']
        elif 'partial_obs' in inputs:
            x = inputs['partial_obs']
        else:
            raise KeyError("inputs must contain 'image' or 'partial_obs'.")

        # Gère les batchs 4D ([N, H, W, C]) ou 5D ([T, B, H, W, C])
        if x.dim() == 5:
            T, B, H, W, C = x.shape
            x = x.view(T * B, H, W, C)
            batch_shape = (T, B)
        elif x.dim() == 4:
            N, H, W, C = x.shape
            x = x.view(N, H, W, C)
            batch_shape = (N,)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x.permute(0, 3, 1, 2)  # [batch, C, H, W]
        x = x.float()
        x = self.feat_extract(x)
        x = x.view(x.shape[0], -1)

        # Traitement du son
        x_sound = inputs['sound']
        if x_sound.dim() == 3:
            x_sound = x_sound.view(-1, x_sound.shape[-1])
        x_sound = x_sound.float()
        x_sound = self.sound_mlp(x_sound)

        # Fusion
        x = torch.cat([x, x_sound], dim=1)
        x = self.fc(x)

        # Sorties
        if len(batch_shape) == 2:
            T, B = batch_shape
            policy_logits = self.policy(x).view(T, B, -1)
            baseline = self.baseline(x).view(T, B)
            binary_output = self.binary_classifier(x).view(T, B)
            action = torch.multinomial(F.softmax(policy_logits.view(-1, self.num_actions), dim=1), 1) if self.training \
                else torch.argmax(policy_logits, dim=2)
            action = action.view(T, B)
        else:
            N = batch_shape[0]
            policy_logits = self.policy(x).view(N, -1)
            baseline = self.baseline(x).view(N)
            binary_output = self.binary_classifier(x).view(N)
            action = torch.multinomial(F.softmax(policy_logits, dim=1), 1) if self.training \
                else torch.argmax(policy_logits, dim=1)
            action = action.view(N)

        return dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            binary_class_output=binary_output
        ), core_state
    

class MinigridPolicyNet_Sound(nn.Module):
    def __init__(self, observation_dict, num_actions):
        super(MinigridPolicyNet_Sound, self).__init__()
        self.observation_shape = observation_dict["image"].shape
        self.observation_shape_sound=observation_dict["sound"].shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.sound_mlp=nn.Sequential(
            init_(nn.Linear(self.observation_shape_sound[0], 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 8))
        )
    
        self.fc = nn.Sequential(
            init_(nn.Linear(40, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))


    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['partial_obs']
        #print("Image input shape:", x.shape)

        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        # SOUND 
        x_sound_inputs=inputs['sound'].squeeze(1)
        #print("Sound input shape:", x_sound_inputs.shape)
        x_sound_inputs=x_sound_inputs.float()
        x_sound=self.sound_mlp(x_sound_inputs)
        x_sound=x_sound.view(T * B, -1)

       

        x_tot = torch.cat((x, x_sound), dim=1) # concat 
        #print("Concatenated input shape:", x_tot.shape)

        #print("x shape:", x.shape)
        #print("x_sound shape:", x_sound.shape)

        #print("x_tot shape:", x_tot.shape) 

        core_input = self.fc(x_tot)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state

class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_dict, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_dict["image"].shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
        self.fc = nn.Sequential(
            init_(nn.Linear(32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))


    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['partial_obs']
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_dict, use_sound=False):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.image_shape = observation_dict["image"].shape
        self.use_sound = use_sound

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.image_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        #sound
        if self.use_sound:
            self.sound_shape = observation_dict["sound"].shape
            self.sound_mlp = nn.Sequential(
                init_(nn.Linear(self.sound_shape[0], 128)),
                nn.ReLU(),
                init_(nn.Linear(128, 32))
            )

        
    """def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() / 255.0 
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)

        #if sound
        if self.use_sound and isinstance(inputs, dict) and 'sound' in inputs:
            sound = inputs['sound'].float()
            sound_emb = self.sound_mlp(sound.view(T*B, -1)).view(T, B, -1)
            return torch.cat([state_embedding, sound_emb], dim=-1)

        return state_embedding"""
    def forward(self, inputs):
        # inputs doit être un dict avec 'image' et éventuellement 'sound'
        if isinstance(inputs, dict):
            x = inputs['image']
            sound = inputs.get('sound', None)
        else:
            x = inputs
            sound = None

        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        state_embedding = x.view(T, B, -1)

        if self.use_sound and sound is not None:
            sound_emb = self.sound_mlp(sound.view(T * B, -1).float()).view(T, B, -1)
            return torch.cat([state_embedding, sound_emb], dim=-1)

        return state_embedding

    """def forward(self, inputs):
        # Vérification que 'inputs' est un dictionnaire
        if isinstance(inputs, dict):
            # Traitement des images
            if 'image' in inputs:
                x = inputs['image']
            else:
                raise ValueError("Input dictionary must contain 'image' key.")
            
            # -- [unroll_length x batch_size x height x width x channels]
            T, B, *_ = x.shape

            # -- [unroll_length*batch_size x height x width x channels]
            x = torch.flatten(x, 0, 1)  # Fusionner le temps et le batch.

            x = x.float() / 255.0  # Normalisation de l'image.

            # -- [unroll_length*batch_size x channels x width x height]
            x = x.transpose(1, 3)
            x = self.feat_extract(x)  # Extraction des caractéristiques.

            state_embedding = x.view(T, B, -1)

            # Traitement du son si présent dans les inputs
            if self.use_sound and 'sound' in inputs:
                sound = inputs['sound'].float()
                sound_emb = self.sound_mlp(sound.view(T*B, -1)).view(T, B, -1)
                return torch.cat([state_embedding, sound_emb], dim=-1)

            # Si pas de son, on retourne uniquement l'embedding d'état
            return state_embedding

        else:
            raise TypeError("The input must be a dictionary containing 'image' and 'sound'.")"""


class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions,input_dim=128):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        self.input_dim = input_dim
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.input_dim, 256)), 
            nn.ReLU(),  
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.input_dim, 256)),
            nn.ReLU(),
        )
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(self.input_dim, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions, input_dim=128):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions
        self.input_dim = input_dim 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.input_dim + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, self.input_dim))


    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MarioDoomPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MarioDoomPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0))

        self.core = nn.LSTM(288, 256, 2)

        self.policy = init_(nn.Linear(256, self.num_actions))
        self.baseline = init_(nn.Linear(256, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['frame']
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        core_input = x.view(T * B, -1)
 
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done'].type(torch.ByteTensor)).float()
        if core_input.is_cuda:
            notdone = notdone.cuda()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class MarioDoomStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MarioDoomStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)
        
        return state_embedding


class MarioDoomForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MarioDoomForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions 
            
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(288 + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, 288))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MarioDoomInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MarioDoomInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 288, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

