import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict


class HERGoalConditionedWrapper(gym.Wrapper):
    """Expose a goal-conditioned dict observation compatible with SB3 HER.

    Keeps the environment reward dynamics intact for online interaction and provides
    `compute_reward` for HER replay relabeling.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base_obs_space = self._extract_base_obs_space(env.observation_space)
        self._obs_shape = tuple(base_obs_space.shape)
        self._obs_dtype = np.float32
        self._desired_goal = np.zeros(self._obs_shape, dtype=self._obs_dtype)

        spaces = {
            "observation": Box(low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=self._obs_dtype),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=self._obs_dtype),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=self._obs_dtype),
        }

        if isinstance(env.observation_space, Dict) and "action_mask" in env.observation_space.spaces:
            spaces["action_mask"] = env.observation_space.spaces["action_mask"]

        self.observation_space = Dict(spaces)
        self.action_space = env.action_space

    @staticmethod
    def _extract_base_obs_space(obs_space):
        if isinstance(obs_space, Dict) and "observations" in obs_space.spaces:
            return obs_space.spaces["observations"]
        return obs_space

    @staticmethod
    def _extract_base_obs(obs):
        if isinstance(obs, dict):
            if "observations" in obs:
                return np.asarray(obs["observations"], dtype=np.float32).ravel()
            if "observation" in obs:
                return np.asarray(obs["observation"], dtype=np.float32).ravel()
        return np.asarray(obs, dtype=np.float32).ravel()

    @staticmethod
    def _extract_action_mask(obs):
        if isinstance(obs, dict) and "action_mask" in obs:
            return np.asarray(obs["action_mask"], dtype=np.float32).ravel()
        return None

    def _wrap_obs(self, obs):
        base_obs = self._extract_base_obs(obs)
        wrapped = {
            "observation": base_obs,
            "achieved_goal": base_obs.copy(),
            "desired_goal": self._desired_goal.copy(),
        }
        action_mask = self._extract_action_mask(obs)
        if action_mask is not None:
            wrapped["action_mask"] = action_mask
        return wrapped

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal_arr = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal_arr = np.asarray(desired_goal, dtype=np.float32)

        if achieved_goal_arr.ndim == 1:
            return float(np.allclose(achieved_goal_arr, desired_goal_arr))

        return np.all(np.isclose(achieved_goal_arr, desired_goal_arr), axis=1).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), reward, terminated, truncated, info


