import lib
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make('CartPole-v1')
model = PPO(MlpPolicy, env, verbose=0)

print('Before training')
mean_reward_before_train = lib.evaluate(model, num_episodes=100, render=True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# Train the agent for 10000 steps
model.learn(total_timesteps=10000)

print('After training')
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
lib.evaluate(model, num_episodes=100, render=True)

