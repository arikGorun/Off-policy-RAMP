from __future__ import annotations

import copy
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

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
        run_args = [
            str(binary_path),
            "-o",
            str(domain_path),
            "-f",
            str(problem_path),
            "-s",
            "0",
            "-t",
            str(self.tolerance),
        ]
        with open(output_path, "wt", encoding="utf-8") as plan_file:
            subprocess.run(
                run_args,
                stdout=plan_file,
                stderr=subprocess.PIPE,
                cwd=str(binary_path.parent),
                timeout=self.timeout,
                check=False,
                shell=False,
            )

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
