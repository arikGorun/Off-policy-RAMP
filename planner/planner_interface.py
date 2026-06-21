from __future__ import annotations

from pathlib import Path
from typing import Iterable

from planner.metricff_wrapper import MetricFFPlanner


class PlannerInterface:
    def __init__(self, min_action_observations=4, max_plan_len=6, metric_ff_root=None, allow_heuristic_fallback=False):
        self.min_action_observations = int(min_action_observations)
        self.max_plan_len = int(max_plan_len)
        self.metric_ff = MetricFFPlanner(metric_ff_root=metric_ff_root)
        self.allow_heuristic_fallback = bool(allow_heuristic_fallback)
        self.last_backend = "none"
        self.last_error = None

    @staticmethod
    def _format_error(error) -> str | None:
        if error is None:
            return None
        return f"{type(error).__name__}: {error}"

    @staticmethod
    def _normalize_plan(plan: Iterable[str]) -> list[str]:
        return [str(action).strip() for action in plan if str(action).strip()]

    @staticmethod
    def _heuristic_fallback(learned_model):
        if not learned_model:
            return None
        candidate_actions = []
        for action, action_data in learned_model.items():
            count = float(action_data.get("count", 0))
            effect = action_data.get("mean_effect")
            score = count * float((effect ** 2).sum()) if effect is not None else count
            if count >= 1:
                candidate_actions.append((int(action), score))
        if not candidate_actions:
            return None
        candidate_actions.sort(key=lambda item: item[1], reverse=True)
        best_action = candidate_actions[0][0]
        return [best_action for _ in range(6)]

    def get_diagnostics(self) -> dict[str, str | bool | None]:
        binary_path: Path | None = None
        resolution_error = None
        try:
            binary_path = self.metric_ff.resolve_binary_path()
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            resolution_error = exc

        return {
            "metric_ff_root": str(self.metric_ff.metric_ff_root),
            "metric_ff_binary_path": str(binary_path) if binary_path is not None else None,
            "metric_ff_binary_available": binary_path is not None,
            "metric_ff_resolution_error": self._format_error(resolution_error),
            "allow_heuristic_fallback": self.allow_heuristic_fallback,
            "last_backend": self.last_backend,
            "last_error": self._format_error(self.last_error),
        }

    def try_plan(self, state, goal, learned_model):
        self.last_backend = "none"
        self.last_error = None
        try:
            plan = self.metric_ff.plan(state)
            if plan:
                self.last_backend = "metric_ff"
                self.last_error = None
                return self._normalize_plan(plan)[: self.max_plan_len]
            self.last_error = RuntimeError("Metric-FF returned no plan for the current state.")
        except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
            self.last_error = exc

        if not self.allow_heuristic_fallback:
            self.last_backend = "none"
            return None

        fallback_plan = self._heuristic_fallback(learned_model)
        if fallback_plan is None:
            self.last_backend = "none"
            if self.last_error is None:
                self.last_error = RuntimeError("Heuristic fallback could not derive a plan from the learned model.")
            return None
        self.last_backend = "heuristic"
        return fallback_plan[: self.max_plan_len]

