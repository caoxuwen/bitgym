import random
import numpy as np
import pandas as pd
import trading_env

df = pd.read_csv('dataset/kaggle_data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
# Need serial number and datetime to be the column name.
df.reset_index(inplace=True)
df.rename(columns={'index':'serial_number', 'Timestamp':'datetime'}, inplace=True)

env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Close', 
                       return_transaction=False, feature_names=['Open', 'High', 'Low', 'Close'])

env.reset()
env.render()

state, reward, done, info = env.step(random.randrange(3))
print('********')
print(state)
print('********')
print(reward)
print('********')

### randow choice action and show the transaction detail
for i in range(500):
    break
    print(i)
    state, reward, done, info = env.step(random.randrange(3))
    print(state, reward)
    env.render()
    if done:
        break
env.transaction_details
