#!/usr/bin/env python3

# Modified from OpenAI baseline ppo1/run_mjoco.py
# Need OpenAI baseline.
# Need mpi4py.

from baselines.common import tf_util as U
from baselines import logger
import pandas as pd
import trading_env

def train(training_env, num_timesteps, evaluation_env = None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    pposgd_simple.learn(training_env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=640,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=1.0, lam=0.95, schedule='linear',
        )
    #training_env.close()

def main():
    df = pd.read_csv('../../bitgym/dataset/btc_indexed2.csv')
    env = trading_env.make(env_id='training_v1', obs_data_len=50, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_reward = True,
                           feature_names=['open', 'high', 'low', 'close', 'volume'])
    logger.configure()
    train(env, num_timesteps=1024*1024*5)

if __name__ == '__main__':
    main()
