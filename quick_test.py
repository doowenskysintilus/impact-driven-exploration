import numpy as np
from minigrid.envs.multiroom_sound import MultiRoomEnvSound
import gymnasium as gym

"""env = MultiRoomEnvSound(
        minNumRooms=1,
        maxNumRooms=3,
        maxRoomSize=4,
        render_mode="human"
    )"""

env = gym.make("MiniGrid-MultiRoomEnvSound-N2-S4-v0", render_mode="human")

num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    nb_sons = 0
    nb_images = 0

    print(f"\n--- Épisode {episode+1} ---")
    while not (terminated or truncated):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if np.any(obs["sound"] != 0):
            nb_sons += 1
            print("Son détecté:", obs["sound"])
        if np.any(obs["image"] != 0):
            nb_images += 1

    print("reward total :", total_reward)
    print("Nombre de sons détectés :", nb_sons)
    print("Image détectée:", nb_images)
env.close()