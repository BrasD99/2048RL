from stable_baselines3 import DQN
from envs.default import GameEnv

env = GameEnv(uri='https://2048game.com')
model = DQN("MlpPolicy", env).learn(total_timesteps=1000000)
model.save('chrome_2048_ppo_cnn')