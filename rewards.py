import gymnasium as gym
import numpy as np

class SnakeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, system_type="dense"):
        super().__init__(env)
        self.system_type = system_type.lower()

    def reward(self, reward):
        # reward == 1 (Food), reward == -1 (Death), reward == 0 (Step) 
        
        if self.system_type == "sparse":
            # Reward System A: Only major events
            if reward == 1: return 10.0
            if reward == -1: return -10.0
            return 0.0

        elif self.system_type == "dense":
            # Reward System B: Constant feedback
            if reward == 1: return 10.0
            if reward == -1: return -10.0
            # Small penalty to encourage speed
            return -0.01

        elif self.system_type == "survivalist":
            # Reward System C: Prioritize life over growth
            if reward == 1: return 5.0
            if reward == -1: return -20.0
            return 0.05
        
        return reward