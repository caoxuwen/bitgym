import random
import numpy as np
import trading_env
import pandas as pd
import talib
import collections

# for talib install package from here http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
#$ ./configure
#$ make
#$ sudo make install
# then finally "pip install TA-lib"


class MAAgent:
    def __init__(self):
        self.shortperiod = 20
        self.longperiod = 60
        self.price_data = collections.deque()

    def choice_action(self, state):
        # action=0 -> do nothing
        # action=1 -> buy 1 share
        # action=2 -> sell 1 share
        close = state[3]
        self.price_data.append(close)
        while len(self.price_data) > self.longperiod+1:
            self.price_data.popleft()
        MA_long = talib.MA(np.array(self.price_data),
                           timeperiod=self.longperiod)
        MA_short = talib.MA(np.array(self.price_data),
                            timeperiod=self.shortperiod)
        MA_long_current = MA_long[-1]

        if len(MA_short) < 2:
            return 0
        MA_short_current = MA_short[-1]
        MA_short_prev = MA_short[-2]

        if MA_long_current == np.nan or MA_short_current == np.nan or MA_short_prev == np.nan:
            return 0

        # crossover normally means trend change
        if (MA_short_prev < MA_long_current and MA_short_current >= MA_long_current):
            return 1
        elif (MA_short_prev > MA_long_current and MA_short_current <= MA_long_current):
            return 2
        return 0


class RSIAgent:
    def __init__(self):
        self.period = 14
        self.price_data = collections.deque()

    def choice_action(self, state):
        # action=0 -> do nothing
        # action=1 -> buy 1 share
        # action=2 -> sell 1 share

        close = state[3]
        self.price_data.append(close)
        while len(self.price_data) > self.period+1:
            self.price_data.popleft()
        rsi = talib.RSI(np.array(self.price_data), self.period)
        # print rsi, rsi[-1], self.price_data
        rsi = rsi[-1]
        # not enough data points yet
        if rsi == np.nan:
            return 0
        # sell when overbought, buy when oversold
        if rsi > 80:
            return 2
        elif rsi < 20:
            return 1
        return 0


class BBANDAgent:
    def __init__(self):
        self.period = 20
        self.price_data = collections.deque()

    def choice_action(self, state):
        # action=0 -> do nothing
        # action=1 -> buy 1 share
        # action=2 -> sell 1 share

        close = state[3]
        self.price_data.append(close)
        while len(self.price_data) > self.period+1:
            self.price_data.popleft()

        upper, middle, lower = talib.BBANDS(np.array(self.price_data),
                                            timeperiod=20,
                                            nbdevup=2.0, nbdevdn=2.0,
                                            matype=talib.MA_Type.EMA)
        #print upper, lower, self.price_data

        upper = upper[-1]
        lower = lower[-1]
        # not enough data points yet
        if upper == np.nan or lower == np.nan:
            return 0

        # sell when overbought, buy when oversold
        if close > upper:
            return 2
        elif close < lower:
            return 1
        return 0


if __name__ == '__main__':
    #agent = MAAgent()
    agent = RSIAgent()
    #agent = BBANDAgent()

    # make env
    df = pd.read_csv('dataset/btc_test.csv')
    print(df.describe())

    env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=0, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30,
                           feature_names=['low', 'high', 'open', 'close', 'volume'])

    state = env.reset()
    env.render()

    """
    df = pd.read_csv('dataset/btc_indexed2.csv')
    print(df.describe())
    env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                       df=df, fee=0.003, max_position=5, sample_days = 30, 
                       return_transaction=False, deal_col_name='close',
                       feature_names=['low', 'high',
                                      'open', 'close',
                                      'volume'])
    state = env.reset()
    env.render()
    """

    print state
    total_rewards = 0
    # randow choice action and show the transaction detail
    while True:
        action = agent.choice_action(state[0])
        state, reward, done, info = env.step(action)
        print state
        total_rewards += reward
        if action != 0 or reward != 0.0:
            print action, reward, total_rewards, done
        env.render()
        if done:
            break

# state, reward, done, info = env.step(2)
# print i,reward
