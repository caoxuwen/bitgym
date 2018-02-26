import random
import numpy as np
import pandas as pd
import trading_env

#df = pd.read_hdf('dataset/SGXTW.h5', 'STW')
#df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df = pd.read_csv('dataset/btc_indexed.csv')
print(df.describe())

env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='close', 
                       feature_names=['low', 'high', 
                                      'open','close', 
                                      'volume','datetime'])

env.reset()
env.render()

state, reward, done, info = env.step(random.randrange(3))

### randow choice action and show the transaction detail
for i in range(2000):
    print(i)
    state, reward, done, info = env.step(random.randrange(3))
    print(state, reward)
    env.render()
    if done:
        break
env.transaction_details