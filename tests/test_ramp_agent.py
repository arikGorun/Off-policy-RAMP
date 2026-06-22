import numpy as np

from aml.action_model_learner import NumericActionModelLearner
from planner.planner_interface import PlannerInterface
from ramp.ramp_agent import RAMPAgent


class DummyActionSpace:
    n = 2

    def sample(self):
        return 0


class DummyEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
        self._step = 0

    def reset(self):
        self._step = 0
        return [0.0, 0.0], {}

    def step(self, action):
        self._step += 1
        done = self._step >= 2
        reward = 1.0 if done else 0.0
        state = [float(self._step), float(action)]
        return state, reward, done, False, {}


class DummyRLAgent:
    def __init__(self):
        self.learn_calls = 0

    def predict(self, state, deterministic=False):
        return 1, None

    def learn(self, total_timesteps=0, reset_num_timesteps=False):
        self.learn_calls += 1
        return self

    def observe_trajectory(self, trajectory):
        return None


class MaskedEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
        self._step = 0

    def reset(self):
        self._step = 0
        return {"action_mask": np.array([1.0, 0.0], dtype=np.float32), "obs": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        self._step += 1
        done = self._step >= 1
        state = {"action_mask": np.array([1.0, 0.0], dtype=np.float32), "obs": np.array([float(self._step)], dtype=np.float32)}
        return state, 0.0, done, False, {}


class AlwaysInvalidRLAgent:
    def predict(self, state, deterministic=False):
        return 1, None

    def learn(self, total_timesteps=0, reset_num_timesteps=False):
        return self

    def observe_trajectory(self, trajectory):
        return None


def test_ramp_agent_collects_metrics():
    env = DummyEnv()
    rl_agent = DummyRLAgent()
    aml = NumericActionModelLearner()
    planner = PlannerInterface(min_action_observations=1, max_plan_len=1)

    agent = RAMPAgent(env=env, rl_agent=rl_agent, aml=aml, planner=planner)
    metrics = agent.train(episodes=5, rl_train_steps=10)

    assert len(metrics.success_rate_ma25) == 5
    assert len(metrics.cumulative_solution_length) == 5
    assert len(metrics.episode_planner_errors) == 5
    assert rl_agent.learn_calls == 5
    assert len(agent.trajectories) == 5


def test_ramp_agent_masks_invalid_rl_actions():
    env = MaskedEnv()
    rl_agent = AlwaysInvalidRLAgent()
    aml = NumericActionModelLearner()
    planner = PlannerInterface(min_action_observations=1000, max_plan_len=1)

    agent = RAMPAgent(env=env, rl_agent=rl_agent, aml=aml, planner=planner)
    outcome = agent.run_episode(train_rl=False)

    assert outcome["steps"] == 1
    assert agent.trajectories
    assert agent.trajectories[0][0][1] == 0


def test_planner_interface_reports_metric_ff_resolution_status():
    planner = PlannerInterface(min_action_observations=1, max_plan_len=1, allow_heuristic_fallback=False)

    diagnostics = planner.get_diagnostics()

    if diagnostics["metric_ff_binary_available"]:
        assert diagnostics["metric_ff_binary_path"] is not None
        assert diagnostics["metric_ff_resolution_error"] is None
    else:
        assert diagnostics["metric_ff_binary_path"] is None
        assert "Metric-FF binary not found" in diagnostics["metric_ff_resolution_error"]


