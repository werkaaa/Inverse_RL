import gym
import numpy as np

from stable_baselines3 import SAC

# env = gym.make("Humanoid-v4")
# env.reset(seed=69)

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000, log_interval=4, progress_bar=True)
# model.save("humanoid")

# # del model  # remove to demonstrate saving and loading

env = gym.make("Humanoid-v4", render_mode="human")
env.reset(seed=42)
model = SAC.load("humanoid", env)

NUM_SAMPLES = 10
# STEPS = 100

# # Evaluate the agent
for i_sample in range(NUM_SAMPLES):

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
            obs = vec_env.reset()

    print(f"Sample {i_sample}", reward)
