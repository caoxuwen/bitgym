import random
import numpy as np
import pandas as pd
import trading_env

#np.set_printoptions(threshold=np.nan)

#df = pd.read_hdf('dataset/SGXTW.h5', 'STW')
#df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df = pd.read_csv('dataset/btc_indexed.csv')
print(df.describe())

env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                       df=df, fee=0.0, max_position=5, deal_col_name='close', 
                       feature_names=['low', 'high', 
                                      'open','close', 
                                      'volume','datetime'])

env.reset()
env.render()

state, reward, done, info = env.step(2)
print(state)
print(state.shape, info)

### randow choice action and show the transaction detail
for i in range(10):
    state, reward, done, info = env.step(0)
    print(state, reward)
    env.render()
    if done:
        break

state, reward, done, info = env.step(2)
print(state, reward)
