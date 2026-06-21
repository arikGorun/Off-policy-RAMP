from pathlib import Path
import logging

from numeric_pddl_gym import MinecraftEnv

from envs.her_goal_conditioned_wrapper import HERGoalConditionedWrapper
from envs.numeric_pddl_wrapper import NumericPDDLWrapper

logger = logging.getLogger(__name__)


DIFFICULTY_TO_FOLDER = {
    "easy": "small",
    "medium": "med",
    "hard": "hard",
}


def build_wooden_sword_env(
    workspace_root: Path,
    difficulty: str = "easy",
    max_steps: int = 250,
    map_size: int = 6,
    masking_strategy: str = "post",
    goal_conditioned: bool = False,
):
    logger.info(f"Building wooden-sword environment (difficulty={difficulty}, map_size={map_size})")
    examples_dir = workspace_root / "NumericPDDLGym" / "examples" / "wooden_sword"
    domain_path = examples_dir / "wooden_sword_domain.pddl"
    problems_dir = examples_dir / DIFFICULTY_TO_FOLDER[difficulty]
    problem_paths = sorted(problems_dir.glob("*.pddl"))

    config = {
        "domain_path": domain_path,
        "problems_list": problem_paths,
        "max_steps": max_steps,
        "executing_algorithm": "RAMP",
        "masking_strategy": masking_strategy,
        "count_inapplicable": False,
        "map_size": map_size,
    }
    logger.info(f"Creating MinecraftEnv with {len(problem_paths)} problem(s)")
    env = NumericPDDLWrapper(MinecraftEnv(config))
    if goal_conditioned:
        logger.info("Wrapping environment as HER goal-conditioned")
        env = HERGoalConditionedWrapper(env)
    logger.info("Wooden-sword environment ready")
    return env
