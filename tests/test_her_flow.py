import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from aml.action_model_learner import NumericActionModelLearner
from planner.planner_interface import PlannerInterface
from ramp.ramp_agent import RAMPAgent
from rl.her_agent import build_her


class TinyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 4
        obs = np.array([float(self._step), float(action)], dtype=np.float32)
        reward = 1.0 if terminated else 0.0
        return obs, reward, terminated, False, {}


def test_her_requires_goal_conditioned_env():
    env = TinyEnv()

    with pytest.raises(RuntimeError, match="HER requires a goal-conditioned environment"):
        build_her(env, verbose=0)


def test_ramp_with_rainbow_like_components_runs_without_her_fallback():
    env = TinyEnv()

    class TinyRLAgent:
        def predict(self, state, deterministic=False):
            return 0, None

        def learn(self, total_timesteps=0, reset_num_timesteps=False):
            return self

    aml = NumericActionModelLearner()
    planner = PlannerInterface(min_action_observations=1000, max_plan_len=1, allow_heuristic_fallback=False)

    agent = RAMPAgent(env=env, rl_agent=TinyRLAgent(), aml=aml, planner=planner)
    metrics = agent.train(episodes=2, rl_train_steps=5)

    assert len(metrics.success_rate_ma25) == 2


