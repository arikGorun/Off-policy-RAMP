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


def build_domain_env(
    workspace_root: Path,
    domain: str,
    difficulty: str = "easy",
    max_steps: int = 250,
    map_size: int = 6,
    masking_strategy: str = "post",
    goal_conditioned: bool = False,
):
    logger.info(f"Building domain environment (domain={domain}, difficulty={difficulty}, map_size={map_size})")
    examples_dir = workspace_root / "NumericPDDLGym" / "examples" / domain
    domain_path = examples_dir / f"{domain}_domain.pddl"
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
    logger.info("Domain environment ready")
    return env


def build_wooden_sword_env(
    workspace_root: Path,
    difficulty: str = "easy",
    max_steps: int = 250,
    map_size: int = 6,
    masking_strategy: str = "post",
    goal_conditioned: bool = False,
):
    return build_domain_env(
        workspace_root=workspace_root,
        domain="wooden_sword",
        difficulty=difficulty,
        max_steps=max_steps,
        map_size=map_size,
        masking_strategy=masking_strategy,
        goal_conditioned=goal_conditioned,
    )

