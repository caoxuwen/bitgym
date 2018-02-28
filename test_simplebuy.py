import random
import numpy as np
import pandas as pd
import trading_env

# np.set_printoptions(threshold=np.nan)

#df = pd.read_hdf('dataset/SGXTW.h5', 'STW')
#df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df = pd.read_csv('dataset/btc_indexed2.csv')
print(df.describe())

env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                       df=df, fee=0.0, max_position=5, deal_col_name='close',
                       sample_days=1,
                       feature_names=['low', 'high',
                                      'open', 'close',
                                      'volume', 'datetime'])

env.reset()
env.render()

state, reward, done, info = env.step(1)
print state

# randow choice action and show the transaction detail
while True:
    state, reward, done, info = env.step(0)
    env.render()
    if done:
        print state, reward
        break
