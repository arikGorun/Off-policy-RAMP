import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from envs.her_goal_conditioned_wrapper import HERGoalConditionedWrapper
from rl.her_agent import build_her


class TinyMaskedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Dict(
            {
                "action_mask": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "observations": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return {
            "action_mask": np.array([1.0, 1.0], dtype=np.float32),
            "observations": np.zeros((3,), dtype=np.float32),
        }, {}

    def step(self, action):
        self._step += 1
        done = self._step >= 1
        return {
            "action_mask": np.array([1.0, 1.0], dtype=np.float32),
            "observations": np.ones((3,), dtype=np.float32) * float(self._step),
        }, 0.0, done, False, {}


def test_goal_conditioned_wrapper_exposes_her_keys():
    env = HERGoalConditionedWrapper(TinyMaskedEnv())
    obs, _ = env.reset()

    assert {"observation", "achieved_goal", "desired_goal"}.issubset(obs.keys())
    assert "action_mask" in obs


def test_build_her_uses_native_her_on_goal_conditioned_env():
    env = HERGoalConditionedWrapper(TinyMaskedEnv())
    agent = build_her(env, verbose=0)

    assert getattr(agent, "replay_buffer_class") is HerReplayBuffer

