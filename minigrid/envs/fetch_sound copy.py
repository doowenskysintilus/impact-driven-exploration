from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces

from minigrid.core.world_object_sound import GoalSoundEngine

from os import path

class FetchEnvSound(MiniGridEnv):
    """
    ## Description

    This environment has multiple objects of assorted types and colors. The
    agent receives a sound as part of its observation telling it which
    object to pick up. Picking up the wrong object terminates the episode with
    zero reward.

    ## Mission Space

    - One sound indicates if the agent is the front of the goal
    - Another for every other object


    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Pick up an object    |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Unused               |
    | 6   | done         | Unused               |

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

    1. The agent picks up the correct object.
    2. The agent picks up the wrong object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    N: number of objects to be generated.

    - `MiniGrid-Fetch-5x5-N2-v0`
    - `MiniGrid-Fetch-6x6-N2-v0`
    - `MiniGrid-Fetch-8x8-N3-v0`

    """

    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, **kwargs):
        
        self.numObjs = numObjs
        self.obj_types = ["key", "ball"]

        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.sound_engine = GoalSoundEngine()

        old_observation_space = self.observation_space

        # Build the new observation space, based on the old one
        new_obs_dict = {key:value for key, value in old_observation_space.items()}
        # + Adding the sound features comming from the sound engine
        new_obs_dict["sound"] = self.sound_engine.sound_space

        self.observation_space = spaces.Dict(new_obs_dict)


    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"{color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._rand_elem(self.obj_types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key and ball.".format(
                        objType
                    )
                )

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = f"{self.targetColor} {self.targetType}"

        # Generate the mission string
        self.mission = "get a %s" % descStr
        assert hasattr(self, "mission")

    def gen_obs(self):
        obs = super().gen_obs()
        obs["sound"] = self.sound_engine.play(self)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        #print(f"step: {self.steps_remaining} {action} {reward} {terminated} {truncated}")

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
