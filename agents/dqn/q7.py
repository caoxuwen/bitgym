import gym

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q7 import config
import pandas as pd
import trading_env


class MyDQN(Linear):

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
        with tf.variable_scope(scope, reuse) as ts:
            bn = tf.layers.batch_normalization(
                inputs=layers.flatten(state))
            full1 = layers.fully_connected(
                bn, num_outputs=256)
            full2 = layers.fully_connected(inputs=full1, num_outputs=256)
            full3 = layers.fully_connected(inputs=full2, num_outputs=256)
            out = layers.fully_connected(
                inputs=full3, num_outputs=num_actions, activation_fn=None)

        return out


class MyDQNCNN(Linear):
    """
    CNN
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
        with tf.variable_scope(scope, reuse) as ts:
            #bn = tf.layers.batch_normalization(
            #    inputs=layers.flatten(state))
            # valid padding
            #conv1_1 = tf.layers.conv2d(inputs=state, filters=16, kernel_size=[1, 5])
            #conv1_2 = tf.layers.conv2d(inputs=state, filters=16, kernel_size=[config.state_history, 1])
            #concat = tf.concat(layers.flatten(conv1_1), layers.flatten(conv1_2), 1)
            conv1 = tf.layers.conv2d(inputs=state, filters=16, kernel_size=[1, 1])
            bn = tf.layers.batch_normalization(inputs=layers.flatten(conv1))
            full = layers.fully_connected(bn, num_outputs=256)
            out = layers.fully_connected(
                inputs=full, num_outputs=num_actions, activation_fn=None)

        return out


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    # make env
    df = pd.read_csv('dataset/btc_indexed2.csv')
    env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=False, sample_days=30,
                           feature_names=['low', 'high', 'open', 'close', 'volume'])

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
    model = MyDQNCNN(env, config)
    model.run(exp_schedule, lr_schedule)
