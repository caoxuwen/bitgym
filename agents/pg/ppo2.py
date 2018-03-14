#!/usr/bin/env python3

# Modified from OpenAI baseline ppo2/run_mujoco.py
# Need OpenAI baseline.
#   openai/baselines/ppo2/ppo2.learn needs to return the trained policy.

import argparse
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

import trading_env
import wrapper

from baselines.ppo2 import ppo2
from baselines import bench, logger

def train(num_timesteps, seed):
    from baselines.common import set_global_seeds
    from policies import MlpPolicy, LstmPolicy
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        df = pd.read_csv('dataset/btc_indexed2.csv')
        env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_reward = True,
                           feature_names=['open', 'high', 'low', 'close', 'volume'])
        env = wrapper.LogPriceFilterWrapper(env)
        return env
    #env = DummyVecEnv([make_env])
    env = DummyVecEnv([make_env] * 8)
    #env = VecNormalize(env)


    set_global_seeds(seed)
    #policy = MlpPolicy
    policy = LstmPolicy
    return ppo2.learn(policy=policy, env=env, nsteps=96*30, nminibatches=8,
        lam=0.95, gamma=1.0, noptepochs=10, log_interval=1, save_interval=10,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)

def evaluation(policy, num_times):
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    def make_env():
        df = pd.read_csv('dataset/btc_indexed2_test.csv')
        env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_reward = False,
                           feature_names=['open', 'high', 'low', 'close', 'volume'])
        env = wrapper.LogPriceFilterWrapper(env)
        return env
    env = DummyVecEnv([make_env] * 8)

    rewards = []
    for i in range(num_times):
      episode_reward = np.zeros(8)
      ob = env.reset()
      lstm_state = policy.initial_state
      not_done = [1]*8
      while True:
        action, _, lstm_state, _ = policy.step(ob, lstm_state, not_done)
        ob, reward, done, _ = env.step(action)
        #print(reward)
        #print(reward.shape)
        episode_reward += reward
        print(done)
        if done.all():
          break
        not_done = np.invert(done).astype(int)
      print("evaluation ",i, ": ", episode_reward)
      rewards.append(list(episode_reward))
    print("evaluation: mu:", np.mean(rewards), "std:", np.std(rewards))


def main():
    logger.configure()
    policy = train(num_timesteps=1024*1024*2, seed=np.random.randint(0, 1024))
    evaluation(policy, 2)
    


if __name__ == '__main__':
    main()
