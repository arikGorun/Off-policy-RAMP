import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

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


def test_ramp_aml_uses_her_relabelled_transitions():
    class OneStepEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "achieved_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "desired_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        def _obs(self, step_value: float):
            obs = np.array([step_value, 0.0], dtype=np.float32)
            return {
                "observation": obs,
                "achieved_goal": obs.copy(),
                "desired_goal": np.zeros((2,), dtype=np.float32),
            }

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._obs(0.0), {}

        def step(self, action):
            return self._obs(1.0), 1.0, True, False, {}

    class FakeSamples:
        def __init__(self):
            self.observations = {
                "observation": torch.tensor([[0.25, 0.0]], dtype=torch.float32),
                "achieved_goal": torch.tensor([[0.25, 0.0]], dtype=torch.float32),
                "desired_goal": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            }
            self.next_observations = {
                "observation": torch.tensor([[0.75, 0.0]], dtype=torch.float32),
                "achieved_goal": torch.tensor([[0.75, 0.0]], dtype=torch.float32),
                "desired_goal": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            }
            self.actions = torch.tensor([[1]], dtype=torch.int64)
            self.rewards = torch.tensor([[0.5]], dtype=torch.float32)
            self.dones = torch.tensor([[0.0]], dtype=torch.float32)

    class FakeHerReplayBuffer:
        def size(self):
            return 1

        def sample(self, batch_size, env=None):
            assert batch_size == 1
            return FakeSamples()

    class TinyHERAgent:
        replay_buffer_class = HerReplayBuffer

        def __init__(self):
            self.replay_buffer = FakeHerReplayBuffer()

        def predict(self, state, deterministic=False):
            return 0, None

        def learn(self, total_timesteps=0, reset_num_timesteps=False):
            return self

    class TinyPlanner:
        last_backend = "none"
        last_error = None

        def try_plan(self, state=None, goal=None, learned_model=None):
            return []

    env = OneStepEnv()
    aml = NumericActionModelLearner()
    planner = TinyPlanner()
    agent = RAMPAgent(env=env, rl_agent=TinyHERAgent(), aml=aml, planner=planner)

    outcome = agent.run_episode(train_rl=True, rl_train_steps=1)

    assert outcome["solved"] is True
    assert len(agent.trajectories) == 2
    assert sum(action_stats["count"] for action_stats in aml.model.values()) == 2


