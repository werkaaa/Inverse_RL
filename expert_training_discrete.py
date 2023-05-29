import gym
from stable_baselines3 import DQN

# Create environment
env = gym.make('CartPole-v1')
env.reset(seed=69)

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3)
# Train the agent
model.learn(total_timesteps=100_000, log_interval=4, progress_bar=True)
# Save the agent
model.save("experts/cartpole")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("experts/cartpole")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
