# BitGym

BitGym contains several agents that applies reinforcement learning on bitcoin trading. This is a CS234 course project.

The agents are in /agents folder, we have three types - dqn-based, pg-based and traditional trading strategies. 

To run training example - python dqn/agents/dqn/q7_prepro.py

To run evaluation - change dqn/agents/dqn/q7_prepro.py last line from model.run to model.test

To see trading behavior in test - change dqn/agents/dqn/configs/q7_prepro.py render_test = True

This project depends on the [TradingGym](https://github.com/Yvictor/TradingGym) project.


# TradingGym

[![Build Status](https://travis-ci.org/Yvictor/TradingGym.svg?branch=master)](https://travis-ci.org/Yvictor/TradingGym)

TradingGym is a toolkit for training and backtesting the reinforcement learning algorithms. This was inspired by OpenAI Gym and imitated the framework form. Not only traning env but also has backtesting and in the future will implement realtime trading env with Interactivate Broker API and so on.

This training env originally design for tickdata, but also support for ohlc data format. WIP.

### Installation
```
git clone https://github.com/Yvictor/TradingGym.git
cd TradingGym
python setup.py install
```



