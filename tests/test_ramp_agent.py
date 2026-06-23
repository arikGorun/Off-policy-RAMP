import copy

import numpy as np
import pytest

from aml.action_model_learner import NumericActionModelLearner
from planner.planner_interface import PlannerInterface
from planner.metricff_wrapper import MetricFFPlanner
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


# ── AML leakage-fix tests ─────────────────────────────────────────────────────

def test_try_plan_returns_none_when_no_aml_model():
    """Planner must not expose ground-truth domain before any learning."""
    planner = PlannerInterface(min_action_observations=1, max_plan_len=6)
    result = planner.try_plan(state=object(), goal=None, learned_model={})
    assert result is None
    assert planner.last_backend == "none"
    assert planner.last_error is not None


class _FakeFunction:
    """Minimal PDDLFunction stand-in with untyped_representation."""
    def __init__(self, name):
        self.untyped_representation = name


class _AnyNode:
    """Minimal tree node that mirrors AnyNode(id=..., value=..., children=[...])."""
    def __init__(self, id_, value, children=None):
        self.id = id_
        self.value = value
        self._children = list(children or [])

    @property
    def children(self):
        return self._children


class _FakeEffect:
    def __init__(self, op, func_name, magnitude):
        func_node = _AnyNode(id_=f"({func_name} )", value=_FakeFunction(func_name))
        mag_node = _AnyNode(id_=str(float(magnitude)), value=float(magnitude))
        self.root = _AnyNode(id_=op, value=op, children=[func_node, mag_node])


class _FakeAction:
    def __init__(self, name, effects):
        self.name = name
        self.numeric_effects = effects


class _FakeDomain:
    def __init__(self, actions):
        self.actions = {a.name: a for a in actions}


class _FakeActionCall:
    def __init__(self, name):
        self.name = name


class _FakeBaseEnv:
    def __init__(self):
        self.grounded_predicates = [object(), object()]   # n_preds = 2
        func_a = _FakeFunction("(count_a )")
        func_b = _FakeFunction("(count_b )")
        self.grounded_functions = [func_a, func_b]
        # Action 0 → base action "craft"
        self.grounded_actions = [_FakeActionCall("craft")]
        self.domain = _FakeDomain([
            _FakeAction("craft", {
                _FakeEffect("increase", "(count_a )", 1.0),
                _FakeEffect("decrease", "(count_b )", 2.0),
            })
        ])


def test_apply_aml_patches_numeric_effects():
    """_apply_aml_to_domain must update effect magnitudes from the learned model
    using exact integer rounding, not floating-point averages."""
    base_env = _FakeBaseEnv()

    # AML: action 0 ("craft") observed 5 times; state vec = [pred0, pred1, func_a, func_b]
    # Learned fluent deltas: func_a +3, func_b -1  (exact integers after rounding)
    learned_model = {
        0: {
            "mean_effect": np.array([0.0, 0.0, 3.01, -0.99], dtype=np.float32),
            "count": 5,
        }
    }

    patched = MetricFFPlanner._apply_aml_to_domain(
        base_env, learned_model, min_observations=4
    )

    craft = patched.actions["craft"]
    eff_map = {}
    for eff in craft.numeric_effects:
        func_name = eff.root.children[0].value.untyped_representation
        eff_map[func_name] = eff.root

    # count_a: rounded +3 → increase 3.0
    assert eff_map["(count_a )"].value == "increase"
    assert eff_map["(count_a )"].children[1].value == 3.0

    # count_b: rounded -1 → decrease 1.0
    assert eff_map["(count_b )"].value == "decrease"
    assert eff_map["(count_b )"].children[1].value == 1.0


def test_apply_aml_skips_actions_below_min_observations():
    """Actions with too few observations must keep their original effects."""
    base_env = _FakeBaseEnv()
    learned_model = {
        0: {
            "mean_effect": np.array([0.0, 0.0, 99.0, -99.0], dtype=np.float32),
            "count": 1,    # below min_observations=4
        }
    }

    patched = MetricFFPlanner._apply_aml_to_domain(
        base_env, learned_model, min_observations=4
    )

    craft = patched.actions["craft"]
    for eff in craft.numeric_effects:
        mag = eff.root.children[1].value
        assert abs(mag) < 10, f"Magnitude {mag} unexpectedly large; leakage via low-count action"


def test_apply_aml_does_not_mutate_original_domain():
    """Patching must deep-copy the domain; the base_env domain stays unchanged."""
    base_env = _FakeBaseEnv()
    learned_model = {
        0: {
            "mean_effect": np.array([0.0, 0.0, 7.0, -5.0], dtype=np.float32),
            "count": 10,
        }
    }

    original_mags = {}
    for eff in base_env.domain.actions["craft"].numeric_effects:
        fname = eff.root.children[0].value.untyped_representation
        original_mags[fname] = eff.root.children[1].value

    MetricFFPlanner._apply_aml_to_domain(base_env, learned_model, min_observations=4)

    for eff in base_env.domain.actions["craft"].numeric_effects:
        fname = eff.root.children[0].value.untyped_representation
        assert eff.root.children[1].value == original_mags[fname], \
            "Original domain was mutated by _apply_aml_to_domain"


def test_apply_aml_skips_action_when_groundings_disagree():
    """If two groundings of the same abstract action produce different integer
    deltas, the action must NOT be patched (conflict detected)."""
    base_env = _FakeBaseEnv()
    # Add a second grounded action that also maps to "craft" but learned a different delta
    base_env.grounded_actions.append(_FakeActionCall("craft"))

    learned_model = {
        0: {
            "mean_effect": np.array([0.0, 0.0, 3.0, -1.0], dtype=np.float32),
            "count": 5,
        },
        1: {
            # Different integer delta for the same abstract action → conflict
            "mean_effect": np.array([0.0, 0.0, 4.0, -2.0], dtype=np.float32),
            "count": 5,
        },
    }

    patched = MetricFFPlanner._apply_aml_to_domain(
        base_env, learned_model, min_observations=4
    )

    # Effects must remain at their original values (1.0 and 2.0)
    craft = patched.actions["craft"]
    for eff in craft.numeric_effects:
        mag = eff.root.children[1].value
        assert mag in (1.0, 2.0), (
            f"Magnitude {mag} was patched despite grounding conflict; "
            "domain integrity violated"
        )
