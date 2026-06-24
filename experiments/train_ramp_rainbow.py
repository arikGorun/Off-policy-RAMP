import argparse
import csv
from datetime import datetime
import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from aml.action_model_learner import NumericActionModelLearner
from envs.pddl_minecraft_gym import build_domain_env
from planner.planner_interface import PlannerInterface
from ramp.ramp_agent import RAMPAgent
from rl.her_agent import build_her
from rl.ppo_agent import build_ppo
from rl.rainbow_agent import build_rainbow

logger = logging.getLogger(__name__)


def log_planner_diagnostics(planner: PlannerInterface) -> dict[str, str | bool | None]:
    diagnostics = planner.get_diagnostics()
    logger.info(
        "Planner diagnostics: metric_ff_root=%s, metric_ff_binary_path=%s, metric_ff_binary_available=%s, "
        "allow_heuristic_fallback=%s, metric_ff_resolution_error=%s",
        diagnostics["metric_ff_root"],
        diagnostics["metric_ff_binary_path"],
        diagnostics["metric_ff_binary_available"],
        diagnostics["allow_heuristic_fallback"],
        diagnostics["metric_ff_resolution_error"],
    )
    return diagnostics


def build_output_dir(workspace_root: Path, algorithm: str, difficulty: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = workspace_root / "logs" / f"{algorithm}_{difficulty}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics_csv(metrics, output_dir: Path) -> Path:
    csv_path = output_dir / "episode_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "episode",
            "reward",
            "steps",
            "success",
            "success_ma25",
            "cumulative_solution_length",
            "planner_used",
            "planner_plan_found",
            "planner_fallback_to_rl",
            "planner_solved",
            "planner_backend",
            "planner_error",
            "rl_loss",
            "rl_batch_seconds",
            "episode_duration_seconds",
        ])
        for index in range(len(metrics.episode_rewards)):
            writer.writerow([
                index,
                metrics.episode_rewards[index],
                metrics.episode_steps[index],
                metrics.episode_successes[index],
                metrics.success_rate_ma25[index],
                metrics.cumulative_solution_length[index],
                metrics.episode_planner_used[index],
                metrics.episode_planner_plan_found[index],
                metrics.episode_planner_fallback_to_rl[index],
                metrics.episode_planner_solved[index],
                metrics.episode_planner_backends[index],
                metrics.episode_planner_errors[index],
                metrics.episode_rl_losses[index],
                metrics.episode_rl_batch_seconds[index],
                metrics.episode_durations_seconds[index],
            ])
    return csv_path


def plot_metrics(metrics, output_dir: Path) -> list[Path]:
    episodes = np.arange(1, len(metrics.episode_rewards) + 1)
    saved_paths: list[Path] = []

    figures = [
        (
            output_dir / "reward_and_loss.png",
            [
                (metrics.episode_rewards, "Episode reward", "Reward"),
                (metrics.episode_rl_losses, "RL loss per episode", "Loss"),
            ],
        ),
        (
            output_dir / "success_and_steps.png",
            [
                (metrics.success_rate_ma25, "Success moving average (25)", "Success MA25"),
                (metrics.episode_steps, "Episode steps", "Steps"),
            ],
        ),
        (
            output_dir / "timing_and_planner.png",
            [
                (metrics.episode_rl_batch_seconds, "RL batch duration", "Seconds"),
                (metrics.episode_durations_seconds, "Episode duration", "Seconds"),
                (metrics.episode_planner_used, "Planner used (0/1)", "Used"),
                (metrics.cumulative_solution_length, "Cumulative solution length", "Length"),
            ],
        ),
    ]

    for figure_path, series in figures:
        fig, axes = plt.subplots(len(series), 1, figsize=(10, 3 * len(series)), sharex="all")
        axes_list: list[Axes] = [axes] if isinstance(axes, Axes) else list(axes.reshape(-1))
        for axis, (values, title, ylabel) in zip(axes_list, series):
            axis.plot(episodes, values, linewidth=1.8)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes_list[-1].set_xlabel("Episode")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        saved_paths.append(figure_path)

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RAMP with planner-to-RL fallback and no missing-implementation fallbacks on NumericPDDLGym wooden-sword tasks."
    )
    parser.add_argument("--algorithm", choices=["rainbow", "her", "ppo"], default="rainbow")
    parser.add_argument("--domain", choices=["wooden_sword", "pogo_stick"], default="wooden_sword")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--rl-train-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-total-seconds", type=float, default=120.0)
    parser.add_argument("--max-episode-seconds", type=float, default=90.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info("Starting train_ramp_rainbow")
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info(
        f"Algorithm={args.algorithm}, Domain={args.domain}, Difficulty={args.difficulty}, Episodes={args.episodes}, Seed={args.seed}"
    )

    map_size = {"easy": 6, "medium": 10, "hard": 15}[args.difficulty]
    logger.info("Building environment...")
    env = build_domain_env(
        workspace_root=args.workspace_root,
        domain=args.domain,
        difficulty=args.difficulty,
        max_steps=250,
        map_size=map_size,
        masking_strategy="post",
        goal_conditioned=(args.algorithm == "her"),
    )
    logger.info("Environment initialized")

    logger.info(f"Building {args.algorithm.upper()} agent...")
    if args.algorithm == "rainbow":
        rl_agent = build_rainbow(env, device=args.device)
    elif args.algorithm == "her":
        rl_agent = build_her(env, device=args.device)
    else:
        rl_agent = build_ppo(env, device=args.device)
    logger.info("Building AML...")
    aml = NumericActionModelLearner()
    logger.info("Building planner interface...")
    planner = PlannerInterface(allow_heuristic_fallback=False)
    logger.info("All components initialized")
    planner_diagnostics = log_planner_diagnostics(planner)


    agent = RAMPAgent(env=env, rl_agent=rl_agent, aml=aml, planner=planner)

    metrics = agent.train(
        episodes=args.episodes,
        rl_train_steps=args.rl_train_steps,
        max_total_seconds=args.max_total_seconds,
        max_episode_seconds=args.max_episode_seconds,
        show_progress=True,
    )
    output_dir = build_output_dir(args.workspace_root, args.algorithm, args.difficulty, args.output_dir)
    csv_path = write_metrics_csv(metrics, output_dir)
    plot_paths = plot_metrics(metrics, output_dir)
    print(f"requested_device={args.device}")
    print(f"actual_device={getattr(rl_agent, 'device', 'unknown')}")
    print(f"algorithm={args.algorithm}")
    print(f"domain={args.domain}")
    print(f"difficulty={args.difficulty}")
    print(f"episodes={args.episodes}")
    print(f"final_success_ma25={metrics.success_rate_ma25[-1] if metrics.success_rate_ma25 else 0.0:.4f}")
    print(f"cumulative_solution_length={metrics.cumulative_solution_length[-1] if metrics.cumulative_solution_length else 0}")
    print(f"efficiency_episode_90pct={metrics.efficiency_episode_90pct}")
    print(f"rl_backend={metrics.rl_backend}")
    print(f"planner_backend_counts={metrics.planner_backend_counts}")
    print(f"planner_metric_ff_root={planner_diagnostics['metric_ff_root']}")
    print(f"planner_metric_ff_binary_path={planner_diagnostics['metric_ff_binary_path']}")
    print(f"planner_metric_ff_binary_available={planner_diagnostics['metric_ff_binary_available']}")
    print(f"planner_metric_ff_resolution_error={planner_diagnostics['metric_ff_resolution_error']}")
    print(f"planner_allow_heuristic_fallback={planner_diagnostics['allow_heuristic_fallback']}")
    print(f"planner_last_error={planner.get_diagnostics()['last_error']}")
    print(f"terminated_early={metrics.terminated_early}")
    print(f"timeout_reason={metrics.timeout_reason}")
    print(f"metrics_csv={csv_path}")
    print(f"plots={[str(path) for path in plot_paths]}")


if __name__ == "__main__":
    main()
