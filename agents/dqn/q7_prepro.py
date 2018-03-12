import gym

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn

from utils.general import get_logger
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear
from utils.wrappers import PreproWrapper
from utils.preprocess import priceNormalization
import numpy as np
from configs.q7_prepro import config
import pandas as pd
import trading_env


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state

        print state.shape
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################
        """
        with tf.variable_scope(scope, reuse) as ts:
            x = tf.layers.batch_normalization(inputs=layers.flatten(state))
            x = layers.fully_connected(x, num_outputs=256)
            x = tf.layers.batch_normalization(inputs=x)
            x = layers.fully_connected(inputs=x, num_outputs=256)
            x = tf.layers.batch_normalization(inputs=x)
            x = layers.fully_connected(inputs=x, num_outputs=256)
            x = tf.layers.batch_normalization(inputs=x)
            out = layers.fully_connected(
                inputs=x, num_outputs=num_actions, activation_fn=None)
        """

        #"""        
        # print "state", state, tf.shape(state)
        with tf.variable_scope(scope, reuse) as ts:
            lstm_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)
            obs_space = self.env.observation_space.shape
            # print "obs_space", obs_space
            x = tf.reshape(
                state, (tf.shape(state)[0], obs_space[1], config.state_history))
            # x = tf.squeeze(state)
            # print "x", x.shape
            #[None, 13, 50]
            x = tf.unstack(x, axis=2)
            # print "x", len(x)
            x, states = rnn.static_rnn(
                lstm_cell, x, dtype=tf.float32)
            out = layers.fully_connected(
                inputs=x[-1], num_outputs=num_actions, activation_fn=None)
        #"""
        ##############################################################
        ######################## END YOUR CODE #######################
        return out

"""


Use deep Q network for test environment.
"""
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
"""


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    # make env
    df = pd.read_csv('dataset/btc_indexed2_train.csv')
    print(df.describe())

    test_env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=pd.read_csv('dataset/btc_indexed2_test.csv'), fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_price=True,
                           feature_names=['low', 'high', 'open', 'close', 'volume'])

    env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_price=True,
                           feature_names=['low', 'high', 'open', 'close', 'volume'], test_env=test_env)
    # env = PreproWrapper(env, prepro=priceNormalization, shape=(1, 5, 1),
    #                    overwrite_render=False)

    env.reset()
    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
    # model.test()
