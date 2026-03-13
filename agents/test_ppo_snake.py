from gymnasium.envs.registration import register
from gym_snakegame.envs.snake_game import SnakeGameEnv

register(
    id="snakegame-v0",  # give it a versioned ID
    entry_point="gym_snakegame.envs.snake_game:SnakeGameEnv",
) 

import gymnasium as gym
from stable_baselines3 import PPO
import time

model = PPO.load("./ppo_snake_logs/best_model.zip")  # use best_model, not final

env = gym.make("snakegame-v0", render_mode="human")
obs, _ = env.reset()

episode = 0
while episode < 10:  # Watch 10 episodes
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)

    if terminated or truncated:
        episode += 1
        print(f"Episode {episode} | Score: {info}")  # info has real score
        obs, _ = env.reset()

env.close()

