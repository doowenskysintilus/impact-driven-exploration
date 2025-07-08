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
import os

from os import path
from abc import ABC, abstractmethod

#minisound_path = path.join("..", "minisound")

#processed_sound_path = path.join(minisound_path, "minigrid", "sound", "Processed")
#features_sound_path = path.join("minigrid", "sound", "Features")


# Obtenir le répertoire absolu du fichier courant (fichier.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin vers minisound (on remonte d'un dossier depuis 'core')
minisound_path = os.path.normpath(os.path.join(current_dir, '..'))

# Construire le chemin complet vers 'Processed' et 'Features'
processed_sound_path = os.path.join(minisound_path, "sound", "Processed")
features_sound_path = os.path.join(minisound_path, "minigrid", "sound", "Features")


goal_sound_path = path.join(processed_sound_path, "goal.wav")
goal_view_path = path.join(processed_sound_path, "ting3.wav")
distractor_sound_path = path.join(processed_sound_path, "stone1.wav")
unlocked_door_sound_path = path.join(processed_sound_path, "locked_door1.wav")
open_door_sound_path = path.join(processed_sound_path, "door_open.wav")
closed_door_sound_path = path.join(processed_sound_path, "close_door.wav")

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
        downscale_factor = int(sound.shape[0] / 600)

        sound = (sound - sound.mean()) / sound.std()
        sound = decimate(sound, q = downscale_factor)
        return sound
    
    def play(self, env):
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)

        from minigrid.core.world_object import Goal
        if isinstance(fwd_cell, Goal):
            generated_sound = self.goal_sound
        else:
            generated_sound = self.distractor_sound

        return generated_sound
    
    """
    def play(self, env):
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)

        # Cas FetchEnvSound (avec cible)
        if hasattr(env, "targetColor") and hasattr(env, "targetType"):
            if fwd_cell is None:
                generated_sound = self.no_sound
            elif getattr(fwd_cell, "color", None) == env.targetColor and getattr(fwd_cell, "type", None) == env.targetType:
                generated_sound = self.goal_sound
            else:
                generated_sound = self.distractor_sound
        # Cas MultiRoomEnv (objectif = Goal)
        else:
            from minigrid.core.world_object import Goal
            if isinstance(fwd_cell, Goal):
                generated_sound = self.goal_sound
            else:
                generated_sound = self.distractor_sound

        return generated_sound
    """
    
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
        downscale_factor = int(sound.shape[0] / 600)

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
    
class ProximityDoorSoundEngine(SoundEngine):
    def __init__(self) -> None:
        super().__init__()

        self.open_door_sound = self.preprocess_sound(wavfile.read(open_door_sound_path)[1])
        self.closed_door_sound = self.preprocess_sound(wavfile.read(closed_door_sound_path)[1])
        self.locked_door_sound = self.preprocess_sound(wavfile.read(unlocked_door_sound_path)[1])
        self.goal_view = self.preprocess_sound(wavfile.read(goal_view_path)[1])
        self.goal_sound = self.preprocess_sound(wavfile.read(goal_sound_path)[1])
        self.distractor_sound = self.preprocess_sound(wavfile.read(distractor_sound_path)[1])
        self.no_sound = np.zeros_like(self.open_door_sound)

    def preprocess_sound(self, sound):
        if sound.ndim == 2:
            sound = sound[:, 0]
        #downscale_factor = int(np.ceil(sound.shape[0] / 6000))
        downscale_factor = int(sound.shape[0] / 600)
        sound = (sound - sound.mean()) / (sound.std())
        sound = decimate(sound, q=downscale_factor)
        if sound.shape[0] > 600:
            sound = sound[:600]
        elif sound.shape[0] < 600:
            sound = np.pad(sound, (0, 600 - sound.shape[0]), mode='constant')
        sound = (sound - sound.min()) / (sound.max() - sound.min())
        return sound.astype(np.float32)
    
    def play_unlock_sound(self):
        self.boolean_for_unlocked_door=True

    def play(self, env):
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)
        agent_pos = env.agent_pos

        from minigrid.core.world_object import Door, Goal

        if fwd_cell is None:
            #generated_sound = self.distractor_sound
            generated_sound = self.no_sound
        elif isinstance(fwd_cell, Goal):
            generated_sound = self.goal_view
        elif isinstance(env.grid.get(*agent_pos), Goal):
            return self.goal_sound
        elif isinstance(fwd_cell, Door):
            if getattr(fwd_cell, "is_locked", False):
                generated_sound = self.locked_door_sound
            elif getattr(fwd_cell, "is_open", False):
                generated_sound = self.open_door_sound
            elif getattr(fwd_cell, "is_closed", False):
                generated_sound = self.closed_door_sound
            else:
                generated_sound = self.closed_door_sound
        else:
            #generated_sound = self.no_sound
            generated_sound = self.distractor_sound

        return generated_sound
    
    """def play(self, env):
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)
        from minigrid.core.world_object import Door

        if fwd_cell is None:
            return np.zeros(600, dtype=np.float32)
        elif isinstance(fwd_cell, Door):
            if getattr(fwd_cell, "is_locked", False):
                return np.ones(600, dtype=np.float32) * 0.8
            elif getattr(fwd_cell, "is_open", False):
                return np.ones(600, dtype=np.float32) * 1.0
            elif getattr(fwd_cell, "is_closed", False):
                return np.ones(600, dtype=np.float32) * 0.5
            else:
                return np.zeros(600, dtype=np.float32)
            
        else:
            return np.zeros(600, dtype=np.float32)"""
    
    """def play(self, env):
        return self.no_sound"""

    @property
    def sound_space(self):
        sound_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(600,),
            dtype=np.float32,
        )
        return sound_space

class ProximityColorDoorSoundEngine(SoundEngine):
    def __init__(self) -> None:
        super().__init__()

        self.sounds = {
            "red": {
                "open": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "door_open.wav"))[1]),
                "closed": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "close_door.wav"))[1]),
                "locked": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "locked_door1.wav"))[1]),
            },
            "blue": {
                "open": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "door_open.wav"))[1]),
                "closed": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "close_door.wav"))[1]),
                "locked": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "locked_door1.wav"))[1]),
            },
            "green": {
                "open": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "door_open.wav"))[1]),
                "closed": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "close_door.wav"))[1]),
                "locked": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "locked_door1.wav"))[1]),
            },
            "gray": {
                "open": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "door_open.wav"))[1]),
                "closed": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "close_door.wav"))[1]),
                "locked": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "locked_door1.wav"))[1]),
            },
            "yellow": {
                "open": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "door_open.wav"))[1]),
                "closed": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "close_door.wav"))[1]),
                "locked": self.preprocess_sound(wavfile.read(path.join(processed_sound_path, "locked_door1.wav"))[1]),
            },
    
        }
        #si la couleur n'est pas reconnue
        self.default_sound = np.zeros(600, dtype=np.float32)

    def preprocess_sound(self, sound):
        if sound.ndim == 2:
            sound = sound[:, 0]
        downscale_factor = int(sound.shape[0] / 600)
        sound = (sound - sound.mean()) / (sound.std())
        sound = decimate(sound, q=downscale_factor)
        if sound.shape[0] > 600:
            sound = sound[:600]
        elif sound.shape[0] < 600:
            sound = np.pad(sound, (0, 600 - sound.shape[0]), mode='constant')
        sound = (sound - sound.min()) / (sound.max() - sound.min())
        return sound.astype(np.float32)

    def play_unlock_sound(self):
        self.boolean_for_unlocked_door=True

    def play(self, env):
        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)
        from minigrid.core.world_object import Door

        if fwd_cell is None or not isinstance(fwd_cell, Door):
            return self.default_sound

        color = getattr(fwd_cell, "color", None)
        color_sounds = self.sounds.get(color, None)
        if color_sounds is None:
            return self.default_sound

        if getattr(fwd_cell, "is_locked", False):
            return color_sounds["locked"]
        elif getattr(fwd_cell, "is_open", False):
            return color_sounds["open"]
        else:
            return color_sounds["closed"]

    @property
    def sound_space(self):
        #shape = next(iter(next(iter(self.sounds.values())).values())).shape
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(600,),
            dtype=np.float32,
        )