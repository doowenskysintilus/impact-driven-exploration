from __future__ import annotations
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
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid

from gymnasium import spaces
from minigrid.core.world_object_sound import FarDoorSoundEngine


class UnlockWithButton_SoundEnv(RoomGrid):
     
    """
    ## Description

    The agent has to open a locked door. This environment can be solved without
    relying on language.

    ## Mission Space

    "open the door"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-UnlockWithButton_SoundEnv-v0`

    """
    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 8
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.sound_engine = FarDoorSoundEngine()
        old_observation_space = self.observation_space
        new_obs_dict = {key:value for key, value in old_observation_space.items()}
        # + Adding the sound features comming from the sound engine
        new_obs_dict["sound"] = self.sound_engine.sound_space
        self.observation_space = spaces.Dict(new_obs_dict)

    @staticmethod
    def _gen_mission():
        return "open the door"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Create a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a button to unlock the door
        self.add_object(0, 0, "bouton", door.color, linked_door=door)
        self.add_object(0, 0, "bouton")
        self.add_object(0, 0, "bouton")
        self.place_agent(0, 0)
        
        self.door = door
        self.unlocked_doors = set()
        self.mission = "open the door"

    def gen_obs(self):
        obs = super().gen_obs()
        obs["sound"] = self.sound_engine.play(self)
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        # if a door is unlocked, play the sound of the door opening
        if self.door.is_locked and self.door not in self.unlocked_doors:
            self.sound_engine.play_unlock_sound()
            self.unlocked_doors.add(self.door)

        return obs, reward, terminated, truncated, info
