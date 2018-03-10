import numpy as np
import gym

from gym import spaces

class PreproWrapper(gym.Wrapper):
    def __init__(self, env, prepro):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
        """
        super(PreproWrapper, self).__init__(env)
        self.viewer = None
        self.prepro = prepro


    def _step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info


    def _reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs

    def close(self):
        return

# This is hack... Only append position to the state.
def _filter(state):
  if state.shape[1] > 5:
    return state[:, [0,1,2,3,4,5]]
  else:
    return state


def _ln(state):
  state[:, 0:4] = np.log(state[:, 0:4])
  return state

class LogPriceFilterWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Args:
            env: (gym env)
        """
        super(LogPriceFilterWrapper, self).__init__(env)
        self.viewer = None
        original_shape = env.observation_space.shape
        new_shape = list(original_shape)
        if original_shape[1] > 8:
          new_shape[1] = 6
        self.observation_space = gym.spaces.Box(low= -np.inf,
                                                high= np.inf,
                                                shape= new_shape)


    def _step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = _ln(_filter(obs))
        return self.obs, reward, done, info


    def _reset(self):
        self.obs = _ln(_filter(self.env.reset()))
        return self.obs

    def close(self):
        return
