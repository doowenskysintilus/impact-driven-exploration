from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.obstructedmaze import ObstructedMazeEnv

from minigrid.core.world_object_sound import FarDoorSoundEngine
from gymnasium import spaces
class ObstructedMaze_1Dlhb_sound(ObstructedMazeEnv):
    """
    A blue ball is hidden in a 2x1 maze. A locked door separates
    rooms. Doors are obstructed by a ball and keys are hidden in boxes.
    """

    def __init__(self, key_in_box=True, blocked=True, **kwargs):
        self.key_in_box = key_in_box
        self.blocked = blocked

        super().__init__(num_rows=1, num_cols=2, num_rooms_visited=2, **kwargs)
        self.sound_engine = FarDoorSoundEngine()
        
        old_observation_space = self.observation_space

        # Build the new observation space, based on the old one
        new_obs_dict = {key:value for key, value in old_observation_space.items()}
        # + Adding the sound features comming from the sound engine
        new_obs_dict["sound"] = self.sound_engine.sound_space

        self.observation_space = spaces.Dict(new_obs_dict)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.add_door(
            0,
            0,
            door_idx=0,
            color=self.door_colors[0],
            locked=True,
            key_in_box=self.key_in_box,
            blocked=self.blocked,
        )

        self.obj, _ = self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.place_agent(0, 0)

    def gen_obs(self):
        obs = super().gen_obs()
        obs["sound"] = self.sound_engine.play(self)
        return obs

    

