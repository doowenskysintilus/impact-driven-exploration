import pygame

from minigrid.envs.fetch_sound import FetchEnvSound

env = FetchEnvSound(10, 3)
obs, info = env.reset()

print(obs)