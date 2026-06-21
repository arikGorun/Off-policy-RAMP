import gymnasium as gym
import numpy as np


class NumericPDDLWrapper(gym.Wrapper):
    def flatten_obs(self, obs):
        if isinstance(obs, dict):
            wrapped = {}
            for key, value in obs.items():
                wrapped[key] = np.asarray(value, dtype=np.float32).ravel()
            return wrapped

        return np.asarray(obs, dtype=np.float32).ravel()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.flatten_obs(obs), reward, terminated, truncated, info
