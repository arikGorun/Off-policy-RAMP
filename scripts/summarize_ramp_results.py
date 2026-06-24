from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RUN_PATTERN = re.compile(
    r"^(?P<algorithm>her|ppo|rainbow)_(?P<difficulty>easy|medium|hard)(?:_[^_]*)*_(?P<runid>\d{8}_\d{6})(?:.*)?$"
)
TARGET_EPISODES = 1000


@dataclass(frozen=True)
class RunRecord:
    algorithm: str
    difficulty: str
    runid: str
    folder: Path
    csv_path: Path
    episodes_logged: int
    success_count: int
    success_rate: float
    final_success_ma25: float
    planner_used_count: int
    planner_used_rate: float
    planner_plan_found_count: int
    planner_plan_found_rate: float
    planner_fallback_count: int
    planner_fallback_rate: float
    planner_solved_count: int
    planner_solved_rate: float
    mean_steps: float
    median_steps: float
    mean_reward: float
    last_episode: int
    status: str
    df: pd.DataFrame


def _parse_run_folder(folder: Path) -> Optional[Tuple[str, str, str]]:
    match = RUN_PATTERN.match(folder.name)
    if not match:
        return None
    return match.group("algorithm"), match.group("difficulty"), match.group("runid")


def _load_run(folder: Path) -> Optional[RunRecord]:
    parsed = _parse_run_folder(folder)
    if parsed is None:
        return None

    csv_path = folder / "episode_metrics.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    episodes_logged = int(len(df))
    success_count = int(df["success"].sum()) if "success" in df.columns else 0
    success_rate = float(df["success"].mean()) if "success" in df.columns else float("nan")
    final_success_ma25 = float(df["success_ma25"].iloc[-1]) if "success_ma25" in df.columns else float("nan")
    planner_used_count = int(df["planner_used"].sum()) if "planner_used" in df.columns else 0
    planner_used_rate = float(df["planner_used"].mean()) if "planner_used" in df.columns else float("nan")
    planner_plan_found_count = int(df["planner_plan_found"].sum()) if "planner_plan_found" in df.columns else planner_used_count
    planner_plan_found_rate = float(df["planner_plan_found"].mean()) if "planner_plan_found" in df.columns else planner_used_rate
    planner_fallback_count = int(df["planner_fallback_to_rl"].sum()) if "planner_fallback_to_rl" in df.columns else 0
    planner_fallback_rate = float(df["planner_fallback_to_rl"].mean()) if "planner_fallback_to_rl" in df.columns else float("nan")
    planner_solved_count = int(df["planner_solved"].sum()) if "planner_solved" in df.columns else 0
    planner_solved_rate = float(df["planner_solved"].mean()) if "planner_solved" in df.columns else float("nan")
    mean_steps = float(df["steps"].mean()) if "steps" in df.columns else float("nan")
    median_steps = float(df["steps"].median()) if "steps" in df.columns else float("nan")
    mean_reward = float(df["reward"].mean()) if "reward" in df.columns else float("nan")
    last_episode = int(df["episode"].iloc[-1]) if "episode" in df.columns else episodes_logged - 1
    status = "complete" if episodes_logged >= TARGET_EPISODES else "partial"

    algorithm, difficulty, runid = parsed
    return RunRecord(
        algorithm=algorithm,
        difficulty=difficulty,
        runid=runid,
        folder=folder,
        csv_path=csv_path,
        episodes_logged=episodes_logged,
        success_count=success_count,
        success_rate=success_rate,
        final_success_ma25=final_success_ma25,
        planner_used_count=planner_used_count,
        planner_used_rate=planner_used_rate,
        planner_plan_found_count=planner_plan_found_count,
        planner_plan_found_rate=planner_plan_found_rate,
        planner_fallback_count=planner_fallback_count,
        planner_fallback_rate=planner_fallback_rate,
        planner_solved_count=planner_solved_count,
        planner_solved_rate=planner_solved_rate,
        mean_steps=mean_steps,
        median_steps=median_steps,
        mean_reward=mean_reward,
        last_episode=last_episode,
        status=status,
        df=df,
    )


def _find_latest_runs(logs_dir: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for folder in logs_dir.iterdir():
        if not folder.is_dir():
            continue
        record = _load_run(folder)
        if record is not None:
            records.append(record)
    return records


def _latest_by_combo(records: List[RunRecord]) -> Dict[Tuple[str, str], RunRecord]:
    latest: Dict[Tuple[str, str], RunRecord] = {}
    for record in records:
        key = (record.algorithm, record.difficulty)
        current = latest.get(key)
        if current is None or record.runid > current.runid:
            latest[key] = record
    return latest


def _write_markdown_table(path: Path, df: pd.DataFrame, title: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(str(df.to_markdown(index=False)))
        handle.write("\n")


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _plot_heatmap(ax, pivot: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    import numpy as np

    matrix = pivot.reindex(index=["her", "ppo", "rainbow"], columns=["easy", "medium", "hard"])
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=np.nanmin(data), vmax=np.nanmax(data), cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "n/a" if pd.isna(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", color="white" if pd.notna(value) and value < (np.nanmax(data) + np.nanmin(data)) / 2 else "black", fontsize=9)
    return im


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RAMP training runs with tables and graphs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(r"C:\Users\Arik\Documents\uni\ResearchMethods\logs"),
        help="Directory containing RAMP run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write summary artifacts. Defaults to a timestamped folder under logs/.",
    )
    args = parser.parse_args()

    records = _find_latest_runs(args.logs_dir)
    if not records:
        raise SystemExit(f"No run folders with episode_metrics.csv found under {args.logs_dir}")

    if args.output_dir is None:
        output_dir = args.logs_dir / "summary_latest"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for record in sorted(records, key=lambda r: (r.algorithm, r.difficulty, r.runid)):
        all_rows.append(
            {
                "algorithm": record.algorithm,
                "difficulty": record.difficulty,
                "runid": record.runid,
                "folder": str(record.folder),
                "episodes_logged": record.episodes_logged,
                "success_count": record.success_count,
                "success_rate": record.success_rate,
                "final_success_ma25": record.final_success_ma25,
                "planner_used_count": record.planner_used_count,
                "planner_used_rate": record.planner_used_rate,
                "planner_plan_found_count": record.planner_plan_found_count,
                "planner_plan_found_rate": record.planner_plan_found_rate,
                "planner_fallback_count": record.planner_fallback_count,
                "planner_fallback_rate": record.planner_fallback_rate,
                "planner_solved_count": record.planner_solved_count,
                "planner_solved_rate": record.planner_solved_rate,
                "mean_steps": record.mean_steps,
                "median_steps": record.median_steps,
                "mean_reward": record.mean_reward,
                "last_episode": record.last_episode,
                "status": record.status,
            }
        )

    all_df = pd.DataFrame(all_rows)
    all_df.sort_values(["algorithm", "difficulty", "runid"], inplace=True)
    all_csv = output_dir / "all_runs_summary.csv"
    all_df.to_csv(all_csv, index=False)
    _write_markdown_table(output_dir / "all_runs_summary.md", all_df, "All discovered RAMP runs")

    latest = _latest_by_combo(records)
    latest_rows = []
    for algorithm in ["her", "ppo", "rainbow"]:
        for difficulty in ["easy", "medium", "hard"]:
            record = latest.get((algorithm, difficulty))
            if record is None:
                latest_rows.append(
                    {
                        "algorithm": algorithm,
                        "difficulty": difficulty,
                        "runid": "n/a",
                        "folder": "n/a",
                        "episodes_logged": 0,
                        "success_count": 0,
                        "success_rate": float("nan"),
                        "final_success_ma25": float("nan"),
                        "planner_used_count": 0,
                        "planner_used_rate": float("nan"),
                        "planner_plan_found_count": 0,
                        "planner_plan_found_rate": float("nan"),
                        "planner_fallback_count": 0,
                        "planner_fallback_rate": float("nan"),
                        "planner_solved_count": 0,
                        "planner_solved_rate": float("nan"),
                        "mean_steps": float("nan"),
                        "median_steps": float("nan"),
                        "mean_reward": float("nan"),
                        "last_episode": float("nan"),
                        "status": "missing",
                    }
                )
            else:
                latest_rows.append(
                    {
                        "algorithm": record.algorithm,
                        "difficulty": record.difficulty,
                        "runid": record.runid,
                        "folder": str(record.folder),
                        "episodes_logged": record.episodes_logged,
                        "success_count": record.success_count,
                        "success_rate": record.success_rate,
                        "final_success_ma25": record.final_success_ma25,
                        "planner_used_count": record.planner_used_count,
                        "planner_used_rate": record.planner_used_rate,
                        "planner_plan_found_count": record.planner_plan_found_count,
                        "planner_plan_found_rate": record.planner_plan_found_rate,
                        "planner_fallback_count": record.planner_fallback_count,
                        "planner_fallback_rate": record.planner_fallback_rate,
                        "planner_solved_count": record.planner_solved_count,
                        "planner_solved_rate": record.planner_solved_rate,
                        "mean_steps": record.mean_steps,
                        "median_steps": record.median_steps,
                        "mean_reward": record.mean_reward,
                        "last_episode": record.last_episode,
                        "status": record.status,
                    }
                )

    latest_df = pd.DataFrame(latest_rows)
    latest_df.to_csv(output_dir / "latest_combo_summary.csv", index=False)
    _write_markdown_table(output_dir / "latest_combo_summary.md", latest_df, "Latest run per algorithm/difficulty")

    completed_df = latest_df[latest_df["status"].isin(["complete", "partial"])].copy()
    success_pivot = completed_df.pivot(index="algorithm", columns="difficulty", values="success_rate")
    ma25_pivot = completed_df.pivot(index="algorithm", columns="difficulty", values="final_success_ma25")
    planner_pivot = completed_df.pivot(index="algorithm", columns="difficulty", values="planner_solved_rate")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = _plot_heatmap(axes[0], success_pivot, "Final success rate", fmt=".2f")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = _plot_heatmap(axes[1], ma25_pivot, "Final success MA-25", fmt=".2f")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = _plot_heatmap(axes[2], planner_pivot, "Planner solved episode rate", fmt=".2f")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    heatmap_path = output_dir / "summary_heatmaps.png"
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)

    algo_order = ["her", "ppo", "rainbow"]
    difficulty_order = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(len(algo_order), 1, figsize=(14, 14), constrained_layout=True)
    if len(algo_order) == 1:
        axes = [axes]
    for ax, algorithm in zip(axes, algo_order):
        found = False
        for difficulty in difficulty_order:
            record = latest.get((algorithm, difficulty))
            if record is None:
                continue
            found = True
            label = f"{difficulty} ({record.status}, {record.episodes_logged} eps)"
            ax.plot(record.df["episode"], record.df["success_ma25"], label=label)
        ax.set_title(f"{algorithm.upper()} success MA-25 over episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("MA-25")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(loc="lower right")
    curves_path = output_dir / "ma25_curves_by_algorithm.png"
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)

    report_lines = [
        "# RAMP training summary so far",
        "",
        f"- Logs directory: `{args.logs_dir}`",
        f"- Summary output directory: `{output_dir}`",
        f"- Discovered run folders: {len(records)}",
        "",
        "## Latest run per algorithm/difficulty",
        "",
        latest_df.to_markdown(index=False),
        "",
        "## Available artifacts",
        "",
        f"- `{all_csv.name}`",
        f"- `latest_combo_summary.csv`",
        f"- `summary_heatmaps.png`",
        f"- `ma25_curves_by_algorithm.png`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"summary_dir={output_dir}")
    print(f"all_runs_csv={all_csv}")
    print(f"latest_combo_csv={output_dir / 'latest_combo_summary.csv'}")
    print(f"heatmaps={heatmap_path}")
    print(f"curves={curves_path}")
    print(f"report={output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



