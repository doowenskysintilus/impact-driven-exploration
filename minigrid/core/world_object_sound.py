from minigrid.core.world_object import (
    WorldObj,
    Floor,
    Wall,
    Door, 
    Ball, 
    Box, 
    Key,
    Bouton
)

from gymnasium import spaces

from scipy.signal import decimate
from scipy.io import wavfile


import numpy as np

from os import path
from abc import ABC, abstractmethod

minisound_path = path.join("..", "minisound")

processed_sound_path = path.join(minisound_path, "minigrid", "sound", "Processed")
features_sound_path = path.join("minigrid", "sound", "Features")

goal_sound_path = path.join(processed_sound_path, "goal.wav")
distractor_sound_path = path.join(processed_sound_path, "stone1.wav")
unlocked_door_sound_path = path.join(processed_sound_path, "locked_door1.wav")
open_door_sound_path = path.join(processed_sound_path, "door_open.wav")

class SoundEngine(ABC):
    
    def __init__(self) -> None:
        pass
    @abstractmethod
    def play(self, env):
        pass


class GoalSoundEngine(SoundEngine):

    def __init__(self) -> None:
        super().__init__()
        
        self.goal_sound = self.preprocess_sound(wavfile.read(goal_sound_path)[1])
        self.distractor_sound = self.preprocess_sound(wavfile.read(distractor_sound_path)[1])

        self.no_sound = np.zeros_like(self.goal_sound)
    
    def preprocess_sound(self, sound):

        # keep only one channel
        sound = sound[:, 0]
        downscale_factor = int(sound.shape[0] / 6000)

        sound = (sound - sound.mean()) / sound.std()
        sound = decimate(sound, q = downscale_factor)
        return sound
    
    def play(self, env):
        # Get the position in front of the agent
        fwd_pos = env.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = env.grid.get(*fwd_pos)

        if fwd_cell is None:
            generated_sound = self.no_sound
        elif fwd_cell.color == env.targetColor and fwd_cell.type == env.targetType:
            generated_sound = self.goal_sound
        else:
            generated_sound = self.distractor_sound

        return generated_sound
    
    @property
    def sound_space(self):
        
        # Change when using features
        sound_space = spaces.Box(
            low=0,
            high=1,
            shape=self.goal_sound.shape,
            dtype=np.float64,
        )
        return sound_space

        
class DoorSoundEngine(SoundEngine):
    def __init__(self) -> None:
        super().__init__()

        self.key = self.preprocess_sound(wavfile.read(goal_sound_path)[1])
        self.unlocked_door_sound = self.preprocess_sound(wavfile.read(unlocked_door_sound_path)[1])
        self.locked_door_sound = self.preprocess_sound(wavfile.read(distractor_sound_path)[1])
        self.open_door_sound = self.preprocess_sound(wavfile.read(open_door_sound_path)[1])

        self.no_sound = np.zeros_like(self.key)
        self.boolean_for_unlocked_door = False

    def preprocess_sound(self, sound):

        # keep only one channel
        sound = sound[:, 0]
        downscale_factor = int(sound.shape[0] / 6000)

        sound = (sound - sound.mean()) / sound.std()
        sound = decimate(sound, q = downscale_factor)
        return sound
    
    def play_unlock_sound(self):
        self.boolean_for_unlocked_door=True

    def play(self, env):
        # Get the position in front of the agent
        fwd_pos = env.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = env.grid.get(*fwd_pos)
        if self.boolean_for_unlocked_door:
            generated_sound = self.unlocked_door_sound
            self.boolean_for_unlocked_door = False
        elif fwd_cell and fwd_cell is Door and not Door.is_locked:
            generated_sound = self.open_door_sound
        elif fwd_cell and fwd_cell is Door and Door.is_locked:
            generated_sound = self.locked_door_sound
        elif fwd_cell and fwd_cell is Key:
            generated_sound = self.key
        else:
            generated_sound = self.no_sound

        return generated_sound

    @property
    def sound_space(self):
        
        # Change when using features
        sound_space = spaces.Box(
            low=0,
            high=1,
            shape=self.key.shape,
            dtype=np.float64,
        )
        return sound_space