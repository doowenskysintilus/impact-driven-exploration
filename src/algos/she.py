# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# she.py
# Algorithme d'apprentissage par renforcement profond avec exploration intrinsèque audio-visuelle
# Basé sur MiniGrid, avec ajout d'une récompense intrinsèque calculée par un discriminateur

#python main.py --model she --env MiniGrid-FetchEnvSoundS8N3-v0 --total_frames 30 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005
import logging
import os
import threading
import time
import timeit
import pprint
import lib_platform

import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers, act

MinigridStateEmbeddingNet = models.MinigridStateEmbeddingNet
MinigridInverseDynamicsNet = models.MinigridInverseDynamicsNet

MinigridPolicyNetClassif = models.MinigridPolicyNetClassif


import matplotlib.pyplot as plt
import numpy as np
import json


replay_buffer = []

# calcul de la loss du discriminateur et génération des paires vraies(1)/fausses(0)

def compute_discriminator_loss(discriminator, visual_state, action, audio_state, discriminator_weighted=False):
    """
    Construire un batch de vraies et fausses paires (image, son), 
    calculer la sortie du discriminateur et la loss binaire associée.
    Retourner la loss, les labels cibles, les prédictions et la taille du batch d'origine.
    """
    expected_audio_dim = discriminator.observation_shape_sound[0]
    device = visual_state.device

    # Mise en forme des entrées
    if audio_state.dim() == 3:
        audio_state = audio_state.reshape(-1, audio_state.shape[-1])
    if visual_state.dim() == 4:
        visual_state = visual_state.reshape(-1, *visual_state.shape[2:])

    # Ajustement de la dimension audio si besoin
    if audio_state.shape[-1] > expected_audio_dim:
        audio_state = audio_state[..., :expected_audio_dim]
    elif audio_state.shape[-1] < expected_audio_dim:
        pad = expected_audio_dim - audio_state.shape[-1]
        audio_state = F.pad(audio_state, (0, pad))

    batch_size = audio_state.shape[0]

    # Vraies paires
    visual_state_true = visual_state
    audio_state_true = audio_state
    labels_true = torch.ones(batch_size, 1, device=device)
    is_empty = (audio_state_true.abs().sum(dim=1) < 1e-5).unsqueeze(1)
    labels_true[is_empty] = 0

    # Fausses paires (shuffle)
    idx = torch.randperm(batch_size, device=device)
    for i in range(batch_size):
        while torch.equal(audio_state[idx[i]], audio_state[i]):
            idx[i] = torch.randint(0, batch_size, (1,), device=device)
    audio_state_false = audio_state[idx]
    visual_state_false = visual_state
    labels_false = torch.zeros(batch_size, 1, device=device)

    # Concatène
    visual_state_batch = torch.cat([visual_state_true, visual_state_false], dim=0)
    audio_state_batch = torch.cat([audio_state_true, audio_state_false], dim=0)
    targets = torch.cat([labels_true, labels_false], dim=0)

    # Mise en forme pour le discriminateur
    if visual_state_batch.dim() > 2 and visual_state_batch.shape[1] == 32:
        visual_state_batch = visual_state_batch.view(-1, *visual_state_batch.shape[2:])
    if audio_state_batch.dim() > 2 and audio_state_batch.shape[1] == 32:
        audio_state_batch = audio_state_batch.view(-1, *audio_state_batch.shape[2:])
    if targets.dim() > 1 and targets.shape[1] == 1:
        targets = targets.view(-1)


    inputs = {
        'partial_obs': visual_state_batch,
        'sound': audio_state_batch,
    }

    # Passage dans le discriminateur des inputs
    outputs, _ = discriminator(inputs)
    preds = outputs['binary_class_output']
    if preds.dim() == 2 and preds.shape[1] != 1:
        preds = preds[:, 0]
    if preds.dim() > 1:
        preds = preds.view(-1)
    if targets.dim() > 1:
        targets = targets.view(-1)


    # Vérification de la cohérence des shapes
    assert preds.shape == targets.shape, f"preds shape: {preds.shape}, targets shape: {targets.shape}"
    discrim_loss = F.binary_cross_entropy(preds, targets, reduction='none')
    return discrim_loss, targets, preds, batch_size

# Fonction pour vérifier si une paire (image, son) existe déjà avec un label différent dans le buffer
def pair_exists_with_different_label(buffer, img, audio, label):
    for tr in buffer:
        if np.allclose(tr['partial_obs'], img) and np.allclose(tr['sound'], audio):
            if tr['label'] != label:
                return True
    return False


def learn(actor_model,
          model,
          state_embedding_model,

          batch,
          initial_agent_state, 
          optimizer,
          state_embedding_optimizer, 
 
          scheduler,
          flags,
          frames=None,
          lock=threading.Lock()):
    """
        Effectue une étape d'apprentissage :
        - Calcule la récompense intrinsèque via le discriminateur
        - Met à jour les réseaux (politique, embedding, etc...)
        - Stocke les transitions dans le replay buffer
        - Retourne les statistiques d'apprentissage
    """
    with lock:
        # Récupération des rewards de comptage (pour exploration basée sur la visite d'états)

        count_rewards = torch.ones((flags.unroll_length, flags.batch_size), 
            dtype=torch.float32).to(device=flags.device)
        count_rewards = batch['episode_state_count'][1:].float().to(device=flags.device)

        
        # Compute intrinsic reward from binary classifier output
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Préparation des entrées pour le discriminateur
        visual_state = batch['partial_obs'].to(flags.device)
        action = batch['action'].to(flags.device)
        audio_state = batch['sound'].to(flags.device)

        # Calcul de la loss du discriminateur et récupération des prédictions
        discrim_loss_tensor, targets, preds, batch_size = compute_discriminator_loss(
            model, visual_state, action, audio_state
        )


        discrim_loss = discrim_loss_tensor.mean()
        
        rewards = batch['reward']

        # Calcul de la récompense intrinsèque (sur les vraies paires uniquement)
        # On utilise les prédictions du discriminateur pour calculer la récompense intrinsèque
        # On ne garde que les prédictions des vraies paires (premier batch_size)
        with torch.no_grad():
            predicted_labels = (preds > 0.5).float()
            accuracy = (predicted_labels.cpu() == targets.cpu()).float().mean().item()

            intrinsic_preds = preds[:batch_size]
            intrinsic_rewards = -torch.log(intrinsic_preds)
            intrinsic_rewards *= flags.intrinsic_reward_coef
        bootstrap_value = learner_outputs['baseline'][-1]

        # Découpe du batch RL (on enlève la première frame pour aligner avec les transitions)
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        rewards = batch['reward']
        # Adapter à la découpe du batch RL (on enlève la première frame)
        T, B = rewards.shape  # rewards est déjà batch['reward'][1:]
        intrinsic_rewards = intrinsic_rewards.reshape(-1, B)[1:].reshape(-1)

        if intrinsic_rewards.shape != rewards.shape:
            # Si intrinsic_rewards est [T*B] et rewards est [T, B]
            if intrinsic_rewards.numel() == rewards.numel():
                intrinsic_rewards = intrinsic_rewards.view_as(rewards)
            # Si intrinsic_rewards est [T] et rewards est [T, B]
            elif intrinsic_rewards.shape[0] == rewards.shape[0] and rewards.dim() == 2:
                intrinsic_rewards = intrinsic_rewards.unsqueeze(1).expand_as(rewards)
            else:
                raise RuntimeError(f"Shapes incompatibles: rewards {rewards.shape}, intrinsic_rewards {intrinsic_rewards.shape}")


        # Ajout des transitions au replay buffer (en évitant les doublons contradictoires)
        for i in range(batch['reward'].shape[0]):
            img = batch['partial_obs'][i].cpu().numpy().tolist()
            audio = batch['sound'][i].cpu().numpy().tolist()
            label = targets[i].item() if targets[i].numel() == 1 else targets[i].cpu().numpy().tolist()
            transition = {
                'partial_obs': img,
                'sound': audio,
                'action': batch['action'][i].item() if batch['action'][i].numel() == 1 else batch['action'][i].cpu().numpy().tolist(),
                'reward': batch['reward'][i].item() if batch['reward'][i].numel() == 1 else batch['reward'][i].cpu().numpy().tolist(),
                'done': batch['done'][i].item() if batch['done'][i].numel() == 1 else batch['done'][i].cpu().numpy().tolist(),
                'intrinsic_reward': intrinsic_rewards[i].item() if intrinsic_rewards[i].numel() == 1 else intrinsic_rewards[i].cpu().numpy().tolist(),
                'label': label
            }
            if not pair_exists_with_different_label(replay_buffer, img, audio, label):
                replay_buffer.append(transition)

        # Préparation des sorties pour le calcul V-trace
        # On enlève la dernière frame car elle n'a pas de récompense associée
        # et on aligne avec les transitions du batch RL
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }
        rewards = batch['reward']

        count_rewards = count_rewards
        
        # Calcul de la récompense totale (extrinsèque + intrinsèque)
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        else:            
            total_rewards = rewards + intrinsic_rewards
        
        discounts = (~batch['done']).float() * flags.discounting

        # Calcul des retours V-trace pour la mise à jour de la politique
        # On utilise les logits de la politique comportementale et de la politique cible
        # pour calculer les avantages et les valeurs cibles
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=total_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        # Calcul des différentes losses
        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])
        
        total_loss = pg_loss + baseline_loss + entropy_loss + discrim_loss

        episode_returns = batch['episode_return'][batch['done']]
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'discrim_loss_mean': discrim_loss.item(),
            #'discrim_loss_tensor': discrim_loss_tensor,
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            #'intrinsic_reward': reward,
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
            'mean_count_rewards': torch.mean(count_rewards).item(),

            'mean_sound_features': torch.mean(batch['sound']).item(),
        }

        
        # Rétropropagation et mise à jour des poids
        optimizer.zero_grad()
        state_embedding_optimizer.zero_grad()
 
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(state_embedding_model.parameters(), flags.max_grad_norm)
   
        optimizer.step()
        state_embedding_optimizer.step()

        scheduler.step()

        # Synchronisation du modèle acteur
        actor_model.load_state_dict(model.state_dict())
        return stats


def train(flags):         
    if flags.xpid is None:
        flags.xpid = 'torchbeast-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid,'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)
    
    
    if 'MiniGrid' in flags.env:
        #ajout de la sound policy

        model = MinigridPolicyNetClassif(env.observation_space, env.action_space.n)
        state_embedding_model = MinigridStateEmbeddingNet(
            env.observation_space,
            use_sound=True).to(device=flags.device)

        buffers = create_buffers(env.observation_space, model.num_actions, flags)
            
        model.share_memory()

        initial_agent_state_buffers = []
        for _ in range(flags.num_buffers):
            state = model.initial_state(batch_size=1)
            for t in state:
                t.share_memory_()
            initial_agent_state_buffers.append(state)
            
        actor_processes = []
        if lib_platform.is_platform_windows:
            context_method = "spawn"
        else:
            context_method = "spawn"
        ctx = mp.get_context(context_method)

        print('Using multiprocessing context:', context_method)

        ctx = mp.get_context(context_method)
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()
            
        episode_state_count_dict = dict()
        train_state_count_dict = dict()
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, free_queue, full_queue, model, buffers, 
                    episode_state_count_dict, train_state_count_dict, 
                    initial_agent_state_buffers, flags))
            actor.start()
            actor_processes.append(actor)

            learner_model = MinigridPolicyNetClassif(env.observation_space, env.action_space.n).to(device=flags.device)

        optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)

        state_embedding_optimizer = torch.optim.RMSprop(
            state_embedding_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
    
    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'discrim_loss_mean',
        'mean_rewards',
        'mean_intrinsic_rewards',
        'mean_total_rewards',
        'mean_sound_features',
        'mean_count_rewards',
        #'intrinsic_reward',
        ]
    logger.info('# Step\t%s', '\t'.join(stat_keys))
    frames, stats = 0, {}

    total_intrinsic_reward = 0.0
    total_intrinsic_count = 0


    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats, total_intrinsic_reward, total_intrinsic_count
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, flags, timings)
            stats = learn(model, learner_model, state_embedding_model, 
                          batch, agent_state, optimizer, 
                          state_embedding_optimizer, 
                          scheduler, flags, frames=frames)
            
            #total_intrinsic_reward += stats['intrinsic_reward'].sum().item()
            #total_intrinsic_count += stats['intrinsic_reward'].numel()
            
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)
    
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
            'model.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'state_embedding_model_state_dict': state_embedding_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state_embedding_optimizer_state_dict': state_embedding_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            log.info('After %i frames: loss %f @ %.1f fps. %sStats:\n%s',
                         frames, total_loss, fps, mean_return,
                         pprint.pformat(stats))

    except KeyboardInterrupt:
        return  
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
        
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
    checkpoint(frames)
    #if total_intrinsic_count > 0:
        #print("Moyenne des récompenses intrinsèques sur tout l'entraînement :",
            #total_intrinsic_reward / total_intrinsic_count)
    
    with open("replay_buffer.txt", "w") as f:
        for transition in replay_buffer:
            f.write(json.dumps(transition) + "\n")
    print("Replay buffer enregistré dans replay_buffer.txt")
    
    plogger.close()

