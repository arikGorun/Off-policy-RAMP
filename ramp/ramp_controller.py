from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List


@dataclass
class ExperimentResult:
    difficulty: str
    seed: int
    efficiency_episode_90pct: int | None
    final_success_ma25: float
    final_cumulative_solution_length: int


class RAMPController:
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory

    def run(self, difficulties: Iterable[str], seeds: Iterable[int], episodes: int, rl_train_steps: int = 250) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for difficulty in difficulties:
            for seed in seeds:
                agent = self.agent_factory(difficulty=difficulty, seed=seed)
                metrics = agent.train(episodes=episodes, rl_train_steps=rl_train_steps)
                results.append(
                    ExperimentResult(
                        difficulty=difficulty,
                        seed=seed,
                        efficiency_episode_90pct=metrics.efficiency_episode_90pct,
                        final_success_ma25=metrics.success_rate_ma25[-1] if metrics.success_rate_ma25 else 0.0,
                        final_cumulative_solution_length=metrics.cumulative_solution_length[-1]
                        if metrics.cumulative_solution_length
                        else 0,
                    )
                )
        return results

    @staticmethod
    def summarize(results: List[ExperimentResult]) -> Dict[str, Dict[str, float]]:
        by_difficulty: Dict[str, List[ExperimentResult]] = {}
        for result in results:
            by_difficulty.setdefault(result.difficulty, []).append(result)

        summary: Dict[str, Dict[str, float]] = {}
        for difficulty, difficulty_results in by_difficulty.items():
            efficiency_values = [r.efficiency_episode_90pct for r in difficulty_results if r.efficiency_episode_90pct is not None]
            summary[difficulty] = {
                "success_rate_ma25": mean([r.final_success_ma25 for r in difficulty_results]),
                "cumulative_solution_length": mean([r.final_cumulative_solution_length for r in difficulty_results]),
                "efficiency_episode_90pct": mean(efficiency_values) if efficiency_values else -1.0,
            }
        return summary
