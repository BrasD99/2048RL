from stable_baselines3 import PPO
from envs.default import GameEnv

env = GameEnv(uri='https://2048game.com')
model = PPO("MlpPolicy", env).learn(total_timesteps=10000)
model.save('chrome_2048_ppo_cnn')