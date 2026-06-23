from __future__ import annotations

import copy
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from pddl_plus_parser.exporters import DomainExporter, MetricFFParser, ProblemExporter


class MetricFFPlanner:
    """Small wrapper around a locally available Metric-FF binary.

    The repository ships the Metric-FF source tree extracted from the user's Downloads folder.
    This wrapper looks for a prebuilt `ff`/`ff.exe` binary under that tree or in `METRIC_FF_DIRECTORY`.
    """

    def __init__(
        self,
        metric_ff_root: Optional[Path] = None,
        tolerance: float = 0.1,
        timeout: int = 60,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.metric_ff_root = Path(metric_ff_root) if metric_ff_root is not None else self._default_root()
        self.tolerance = float(tolerance)
        self.timeout = int(timeout)
        self.domain_exporter = DomainExporter()
        self.problem_exporter = ProblemExporter()
        self.parser = MetricFFParser()

    @staticmethod
    def _default_root() -> Path:
        env_root = os.environ.get("METRIC_FF_DIRECTORY")
        if env_root:
            return Path(env_root)
        return Path(__file__).resolve().parents[1] / "external" / "Metric-FF" / "Metric-FF-v2.1"

    def _candidate_binaries(self) -> Iterable[Path]:
        roots = [self.metric_ff_root]
        roots.extend([
            self.metric_ff_root / "bin",
            self.metric_ff_root / "build",
        ])
        for root in roots:
            for binary_name in ("ff.exe", "ff", "ff.cmd", "ff.bat"):
                candidate = root / binary_name
                if candidate.exists():
                    yield candidate

    def resolve_binary_path(self) -> Path:
        for candidate in self._candidate_binaries():
            return candidate
        raise FileNotFoundError(
            "Metric-FF binary not found. Extracted source is present, but the planner must be built to produce `ff` or `ff.exe`. "
            f"Looked under: {self.metric_ff_root}. "
            "Run `scripts/build_metric_ff.ps1 -CheckOnly` to validate your toolchain and "
            "`scripts/build_metric_ff.ps1` to build the binary."
        )

    @staticmethod
    def _unwrap_env(env):
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "domain") and hasattr(current, "current_problem"):
                return current
            current = getattr(current, "env", None)
        raise AttributeError("Could not locate a base PDDL environment with `domain` and `current_problem`.")

    def _build_problem_snapshot(self, env):
        base_env = self._unwrap_env(env)
        problem = copy.deepcopy(base_env.current_problem)
        if hasattr(base_env, "state") and base_env.state is not None:
            problem.initial_state_predicates = copy.deepcopy(base_env.state.state_predicates)
            problem.initial_state_fluents = copy.deepcopy(base_env.state.state_fluents)
        return base_env.domain, problem

    def _write_temp_problem_files(self, env, temp_dir: Path) -> tuple[Path, Path]:
        domain, problem = self._build_problem_snapshot(env)
        domain_path = temp_dir / "metricff_domain.pddl"
        problem_path = temp_dir / "metricff_problem.pddl"
        self.domain_exporter.export_domain(domain, domain_path)
        self.problem_exporter.export_problem(problem, problem_path)
        return domain_path, problem_path

    def _run_metric_ff(self, binary_path: Path, domain_path: Path, problem_path: Path, output_path: Path) -> None:
        if os.name == "nt" and binary_path.suffix.lower() != ".exe":
            self._run_metric_ff_via_wsl(binary_path, domain_path, problem_path, output_path)
            return

        run_args = [
            str(binary_path),
            "-o",
            str(domain_path),
            "-f",
            str(problem_path),
            "-s",
            "0",
        ]
        completed = subprocess.run(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(binary_path.parent),
            timeout=self.timeout,
            check=False,
            shell=False,
            text=True,
        )

        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""

        if completed.returncode != 0:
            self.logger.debug(
                "Metric-FF returned non-zero exit code (%s). stderr=%s",
                completed.returncode,
                stderr_text.strip(),
            )

        output_path.write_text(stdout_text, encoding="utf-8", errors="ignore")

    @staticmethod
    def _windows_to_wsl_path(path: Path) -> str:
        text = str(path).replace("\\", "/")
        match = re.match(r"^([A-Za-z]):/(.+)$", text)
        if not match:
            raise ValueError(f"Cannot convert Windows path to WSL path: {path}")
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"

    def _run_metric_ff_via_wsl(self, binary_path: Path, domain_path: Path, problem_path: Path, output_path: Path) -> None:
        run_args = [
            "wsl.exe",
            "--",
            self._windows_to_wsl_path(binary_path),
            "-o",
            self._windows_to_wsl_path(domain_path),
            "-f",
            self._windows_to_wsl_path(problem_path),
            "-s",
            "0",
        ]
        completed = subprocess.run(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            check=False,
            shell=False,
            text=True,
        )

        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""

        if completed.returncode != 0:
            self.logger.debug(
                "Metric-FF (WSL) returned non-zero exit code (%s). stderr=%s",
                completed.returncode,
                stderr_text.strip(),
            )

        output_path.write_text(stdout_text, encoding="utf-8", errors="ignore")

    def plan(self, env) -> List[str]:
        """Run Metric-FF and return the plan as grounded PDDL action strings.

        The wrapper exports the current environment state into temporary PDDL files, runs the local
        Metric-FF binary, and then parses the returned plan file with `MetricFFParser`.
        """

        binary_path = self.resolve_binary_path()
        with tempfile.TemporaryDirectory(prefix="metricff_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            domain_path, problem_path = self._write_temp_problem_files(env, tmp_path)
            raw_output_path = tmp_path / "metricff_output.txt"
            parsed_plan_path = tmp_path / "metricff_plan.txt"
            self._run_metric_ff(binary_path, domain_path, problem_path, raw_output_path)

            status, actions = self.parser.get_solving_status(raw_output_path)
            if status != "ok":
                return []

            self.parser.parse_plan(raw_output_path, parsed_plan_path)
            if parsed_plan_path.exists():
                actions = [line.strip() for line in parsed_plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return actions

    @staticmethod
    def _apply_aml_to_domain(base_env: Any, learned_model: Dict, min_observations: int):
        """Return a deep-copied domain whose numeric effect magnitudes are patched
        with values from the AML learned model.

        Design contract for **discrete** domains
        ─────────────────────────────────────────
        Every grounding of the same abstract action produces the *same* deterministic
        integer numeric effect (e.g. BREAK always adds exactly 1 log regardless of
        which cell it is applied to).  AML's ``mean_effect`` for a grounding that has
        been observed enough times is therefore a noisy estimate of that integer —
        rounding it recovers the true value.

        We therefore:
        1.  Round each well-observed grounding's fluent delta to the nearest integer.
        2.  Require ALL well-observed groundings of the same abstract action to agree
            on that integer.  If they disagree the abstract action is not updated
            (something is wrong with the learned model).
        3.  Never average across groundings — there is no meaningful "average effect"
            in a deterministic discrete domain.

        Only actions whose observation count meets *min_observations* are considered.

        The AML state vector layout is assumed to be:
            [grounded_predicate_0, ..., grounded_predicate_N-1,
             grounded_function_0, ..., grounded_function_M-1]
        which matches *base_env.grounded_predicates* and *base_env.grounded_functions*.
        """
        logger = logging.getLogger(__name__)

        domain = copy.deepcopy(base_env.domain)
        n_preds = len(base_env.grounded_predicates)
        grounded_funcs = base_env.grounded_functions  # sorted; index = fluent dim offset

        fluent_to_idx: Dict[str, int] = {
            f.untyped_representation: i for i, f in enumerate(grounded_funcs)
        }

        # Build per-abstract-action integer fluent deltas.
        # Each well-observed grounding contributes a rounded integer delta vector.
        # All groundings must agree; the first one seen anchors the expected value.
        action_name_to_int_delta: Dict[str, np.ndarray] = {}
        action_name_conflict: set = set()

        for action_idx, aml_data in learned_model.items():
            if aml_data.get("count", 0) < min_observations:
                continue
            try:
                action_call = base_env.grounded_actions[int(action_idx)]
            except (IndexError, TypeError, ValueError):
                continue

            aname = action_call.name
            if aname in action_name_conflict:
                continue  # already flagged; skip further groundings

            # Round to integer — discrete effects are exact integers, mean_effect
            # is just a noisy estimate from repeated identical observations.
            raw_fluent = np.asarray(aml_data["mean_effect"], dtype=np.float64)[n_preds:]
            int_delta = np.round(raw_fluent).astype(int)

            if aname not in action_name_to_int_delta:
                action_name_to_int_delta[aname] = int_delta
            elif not np.array_equal(action_name_to_int_delta[aname], int_delta):
                # Two groundings of the same abstract action disagree → do not patch.
                logger.warning(
                    "_apply_aml_to_domain: groundings of '%s' disagree on numeric "
                    "fluent deltas (%s vs %s); skipping this action.",
                    aname,
                    action_name_to_int_delta[aname].tolist(),
                    int_delta.tolist(),
                )
                del action_name_to_int_delta[aname]
                action_name_conflict.add(aname)

        # Patch numeric effect nodes in the copied domain
        for aname, action in domain.actions.items():
            if aname not in action_name_to_int_delta:
                continue
            fluent_deltas = action_name_to_int_delta[aname]

            for eff in action.numeric_effects:
                root = eff.root
                children = list(root.children)
                if len(children) < 2:
                    continue
                func_node = children[0]
                mag_node = children[1]

                func_val = func_node.value
                if not hasattr(func_val, "untyped_representation"):
                    continue
                fluent_name = func_val.untyped_representation
                if fluent_name not in fluent_to_idx:
                    continue

                fidx = fluent_to_idx[fluent_name]
                if fidx >= len(fluent_deltas):
                    continue
                learned_delta = int(fluent_deltas[fidx])

                if learned_delta == 0:
                    continue  # no integer effect; leave original

                # Update magnitude and direction in-place on the tree node
                new_mag = float(abs(learned_delta))
                mag_node.value = new_mag
                mag_node.id = str(new_mag)

                if learned_delta > 0:
                    root.value = "increase"
                    root.id = "increase"
                else:
                    root.value = "decrease"
                    root.id = "decrease"

        return domain

    def plan_with_model(self, env, learned_model: Dict, min_observations: int = 4) -> List[str]:
        """Run Metric-FF on a domain patched with AML-learned numeric effects.

        The problem snapshot is still taken from the current env state so that the
        planner always sees the correct initial conditions and goal.
        """
        base_env = self._unwrap_env(env)
        aml_domain = self._apply_aml_to_domain(base_env, learned_model, min_observations)

        _, problem = self._build_problem_snapshot(env)

        binary_path = self.resolve_binary_path()
        with tempfile.TemporaryDirectory(prefix="metricff_aml_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            domain_path = tmp_path / "metricff_domain.pddl"
            problem_path = tmp_path / "metricff_problem.pddl"
            raw_output_path = tmp_path / "metricff_output.txt"
            parsed_plan_path = tmp_path / "metricff_plan.txt"

            self.domain_exporter.export_domain(aml_domain, domain_path)
            self.problem_exporter.export_problem(problem, problem_path)
            self._run_metric_ff(binary_path, domain_path, problem_path, raw_output_path)

            status, actions = self.parser.get_solving_status(raw_output_path)
            if status != "ok":
                return []

            self.parser.parse_plan(raw_output_path, parsed_plan_path)
            if parsed_plan_path.exists():
                actions = [
                    line.strip()
                    for line in parsed_plan_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            return actions
