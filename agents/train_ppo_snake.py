from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import os

# Register custom environment
register(
    id="snakegame-v0",
    entry_point="gym_snakegame.envs.snake_game:SnakeGameEnv",
)

log_dir = "./ppo_snake_logs/"
os.makedirs(log_dir, exist_ok=True)

# Training environment
env = make_vec_env("snakegame-v0", n_envs=8)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
)

# Evaluation environment
eval_env = make_vec_env("snakegame-v0", n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Train
model.learn(total_timesteps=2_000_000, callback=eval_callback)

# Save model + normalization stats
model.save(os.path.join(log_dir, "ppo_snake_final"))
env.save(os.path.join(log_dir, "vec_normalize.pkl"))

print("✅ Training complete and model saved.")