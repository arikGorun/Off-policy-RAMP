from collections import Counter
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


Transition = Tuple[Any, int, float, Any, bool]


@dataclass
class RampMetrics:
    success_rate_ma25: List[float]
    cumulative_solution_length: List[int]
    efficiency_episode_90pct: Optional[int]
    planner_backend_counts: Dict[str, int]
    rl_backend: str
    terminated_early: bool
    timeout_reason: Optional[str]
    episode_rewards: List[float]
    episode_steps: List[int]
    episode_successes: List[int]
    episode_planner_used: List[int]
    episode_planner_plan_found: List[int]
    episode_planner_fallback_to_rl: List[int]
    episode_planner_solved: List[int]
    episode_planner_backends: List[str]
    episode_planner_errors: List[str]
    episode_rl_losses: List[float]
    episode_rl_batch_seconds: List[float]
    episode_durations_seconds: List[float]


class RAMPAgent:
    def __init__(self, env, rl_agent, aml, planner):
        self.env = env
        self.rl_agent = rl_agent
        self.aml = aml
        self.planner = planner
        self.trajectories: List[List[Transition]] = []

    @staticmethod
    def _moving_average_success(successes: Sequence[int], window: int = 25) -> float:
        window_values = successes[-window:]
        return float(sum(window_values)) / float(len(window_values)) if window_values else 0.0

    @staticmethod
    def _extract_action_mask(state: Any) -> Optional[np.ndarray]:
        if not isinstance(state, dict):
            return None

        if "action_mask" in state:
            mask = np.asarray(state["action_mask"], dtype=np.float32).ravel()
            return mask if mask.size else None

        return None

    @staticmethod
    def _to_int_action(action: Any) -> int:
        action_arr = np.asarray(action)
        if action_arr.size == 1:
            return int(action_arr.reshape(-1)[0])
        return int(action)

    @staticmethod
    def _canonical_action_text(action_text: str) -> str:
        text = str(action_text).strip().lower()
        text = text.replace("(", " ( ").replace(")", " ) ")
        return " ".join(text.split())

    def _extract_model_for_action_scoring(self):
        return getattr(self.rl_agent, "model", self.rl_agent)

    def _extract_model_for_logging(self):
        return getattr(self.rl_agent, "model", self.rl_agent)

    def _get_latest_rl_loss(self) -> float:
        model = self._extract_model_for_logging()
        logger_values = getattr(getattr(model, "logger", None), "name_to_value", None)
        if not logger_values:
            return float("nan")

        loss_value = logger_values.get("train/loss")
        if loss_value is None:
            return float("nan")
        return float(loss_value)

    def _get_planner_error(self) -> str:
        error = getattr(self.planner, "last_error", None)
        if error is None:
            return ""
        return f"{type(error).__name__}: {error}"

    def _get_action_scores(self, state: Any) -> Optional[np.ndarray]:
        model = self._extract_model_for_action_scoring()
        if not (hasattr(model, "policy") and hasattr(model, "q_net")):
            return None

        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(state)
            q_values = model.q_net(obs_tensor)

        q_values_arr = q_values.detach().cpu().numpy()
        if q_values_arr.ndim == 2 and q_values_arr.shape[0] > 0:
            return np.asarray(q_values_arr[0], dtype=np.float32)
        return np.asarray(q_values_arr, dtype=np.float32).reshape(-1)

    @staticmethod
    def _mask_and_renormalize(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = (mask > 0.0).astype(np.float32)
        if valid.sum() <= 0.0:
            raise ValueError("Action mask has no valid actions to sample from.")

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != valid.shape[0]:
            raise ValueError("Action scores and mask must have same size.")

        shifted_scores = scores - float(np.max(scores))
        probs = np.exp(shifted_scores)
        probs *= valid
        prob_sum = float(probs.sum())

        if prob_sum <= 0.0:
            valid_count = int(np.count_nonzero(valid))
            probs = np.zeros_like(valid, dtype=np.float32)
            probs[valid > 0.0] = 1.0 / float(valid_count)
            return probs

        return np.asarray(probs / prob_sum, dtype=np.float32)

    @staticmethod
    def _unwrap_env_for_attr(env: Any, attr_name: str) -> Any:
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, attr_name):
                return current
            current = getattr(current, "env", None)
        return None

    def _match_plan_to_action_index(self, env_obj: Any, planned_action: Any) -> Optional[int]:
        if not (hasattr(env_obj, "action_space") and hasattr(env_obj, "get_action_from_rl")):
            return None

        action_count = int(getattr(getattr(env_obj, "action_space", None), "n", 0))
        if action_count <= 0:
            return None

        target = self._canonical_action_text(str(planned_action))
        for action_index in range(action_count):
            try:
                grounded = env_obj.get_action_from_rl(action_index)
            except Exception:
                continue
            if self._canonical_action_text(str(grounded)) == target:
                return int(action_index)

        return None

    def _resolve_action(self, state: Any, plan: List[Any]) -> int:
        logger.debug(f"_resolve_action: state={type(state)}, plan_len={len(plan)}, plan={plan[:3] if len(plan) > 3 else plan}")

        if plan:
            logger.debug(f"_resolve_action: plan available, popping first action")
            planned_action = plan.pop(0)
            logger.debug(f"_resolve_action: popped planned_action={planned_action}, type={type(planned_action)}")

            if isinstance(planned_action, int):
                logger.debug(f"_resolve_action: planned_action is already int, returning {planned_action}")
                return int(planned_action)

            logger.debug(f"_resolve_action: planned_action is not int, attempting conversion/mapping")

            rl_mapping_env = self._unwrap_env_for_attr(self.env, "get_action_from_rl")
            if rl_mapping_env is not None:
                resolved_idx = self._match_plan_to_action_index(rl_mapping_env, planned_action)
                if resolved_idx is not None:
                    logger.debug(f"_resolve_action: matched planner action string to RL action index {resolved_idx}")
                    return resolved_idx

            planning_env = self._unwrap_env_for_attr(self.env, "get_action_from_planning")
            if planning_env is not None:
                logger.debug(f"_resolve_action: found env with get_action_from_planning ({type(planning_env)})")
                grounded_action = planning_env.get_action_from_planning(str(planned_action))
                logger.debug(f"_resolve_action: grounded_action={grounded_action}")

                if isinstance(grounded_action, (int, np.integer)):
                    resolved = int(grounded_action)
                    logger.debug(f"_resolve_action: planner returned integer action id {resolved}")
                    return resolved

                grounded_key = str(grounded_action)
                grounded_map = getattr(planning_env, "grounded_actions_map", {})
                logger.debug(f"_resolve_action: grounded_map size={len(grounded_map)}")

                if grounded_key in grounded_map:
                    resolved = int(grounded_map[grounded_key])
                    logger.debug(f"_resolve_action: found in grounded_map, resolved to {resolved}")
                    return resolved

                if hasattr(planning_env, "grounded_actions"):
                    logger.debug(f"_resolve_action: trying grounded_actions index lookup")
                    resolved = int(planning_env.grounded_actions.index(grounded_action))
                    logger.debug(f"_resolve_action: found in grounded_actions at index {resolved}")
                    return resolved

            logger.debug(f"_resolve_action: no mapping found, fallback to direct int conversion")
            return int(planned_action)

        action_mask = self._extract_action_mask(state)
        if action_mask is None:
            logger.debug("_resolve_action: no action mask found, delegating to RL agent.predict()")
            action, _ = self.rl_agent.predict(state, deterministic=False)
            resolved_action = self._to_int_action(action)
            logger.debug(f"_resolve_action: RL agent predicted action={resolved_action}")
            return resolved_action

        valid_actions = np.flatnonzero(action_mask > 0.0)
        logger.debug(f"_resolve_action: extracted action mask with {valid_actions.size} valid action(s)")

        action_scores = self._get_action_scores(state)
        if action_scores is None:
            logger.debug("_resolve_action: action scores unavailable, using uniform distribution over valid actions")
            masked_probs = action_mask.astype(np.float32)
            masked_probs /= float(masked_probs.sum())
        else:
            masked_probs = self._mask_and_renormalize(action_scores, action_mask)
            logger.debug(
                f"_resolve_action: masked and renormalized probabilities ready "
                f"(max_prob={float(masked_probs.max()):.4f})"
            )

        candidate_actions = np.arange(masked_probs.shape[0], dtype=np.int64)
        sampled_action = int(np.random.choice(candidate_actions, p=masked_probs))
        logger.debug(f"_resolve_action: sampled masked action={sampled_action}")
        return sampled_action

    def run_episode(self, train_rl: bool = True, rl_train_steps: int = 250, max_episode_seconds: Optional[float] = None) -> Dict[str, Any]:
        logger.info(f"run_episode: starting (train_rl={train_rl}, rl_train_steps={rl_train_steps})")
        logger.info("run_episode: resetting environment")
        reset_start = time.perf_counter()
        state, _ = self.env.reset()
        reset_elapsed = time.perf_counter() - reset_start
        logger.info(f"run_episode: env.reset() completed in {reset_elapsed:.3f}s, state type={type(state)}")

        logger.info("run_episode: attempting planner.try_plan()")
        planner_start = time.perf_counter()
        plan = self.planner.try_plan(state=self.env, goal=None, learned_model=getattr(self.aml, "model", {})) or []
        planner_elapsed = time.perf_counter() - planner_start
        planner_plan_found = bool(plan)
        planner_error = self._get_planner_error()
        logger.info(
            f"run_episode: planner.try_plan() completed in {planner_elapsed:.3f}s, "
            f"planner_plan_found={planner_plan_found}, plan_len={len(plan)}, backend={getattr(self.planner, 'last_backend', 'unknown')}, "
            f"error={planner_error or 'none'}"
        )

        episode_start = time.perf_counter()

        trajectory: List[Transition] = []
        done = False
        solved = False
        total_reward = 0.0
        steps = 0
        planner_action_steps = 0
        rl_action_steps = 0
        planner_fallback_to_rl = False

        logger.info(f"run_episode: entering step loop, max_episode_seconds={max_episode_seconds}")
        while not done:
            elapsed_in_episode = time.perf_counter() - episode_start
            if max_episode_seconds is not None and elapsed_in_episode > max_episode_seconds:
                logger.error(f"run_episode: TIMEOUT in step loop at step {steps}, elapsed={elapsed_in_episode:.1f}s > limit {max_episode_seconds:.1f}s")
                raise TimeoutError(f"Episode exceeded time limit of {max_episode_seconds:.1f}s")

            logger.debug(f"run_episode: step {steps}, plan_len={len(plan)}, resolved_action starting")
            action_start = time.perf_counter()
            used_planner_action = bool(plan)
            action = self._resolve_action(state, plan)
            action_elapsed = time.perf_counter() - action_start
            logger.debug(f"run_episode: step {steps}, action resolved to {action} in {action_elapsed:.3f}s")

            if used_planner_action:
                planner_action_steps += 1
            else:
                rl_action_steps += 1
                if planner_plan_found:
                    planner_fallback_to_rl = True

            logger.debug(f"run_episode: step {steps}, calling env.step({action})")
            step_start = time.perf_counter()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            step_elapsed = time.perf_counter() - step_start
            logger.debug(f"run_episode: step {steps}, env.step() completed in {step_elapsed:.3f}s, reward={reward}, terminated={terminated}, truncated={truncated}")

            done = bool(terminated or truncated)
            solved = solved or bool(terminated and reward > 0)
            total_reward += float(reward)
            steps += 1

            trajectory.append((state, action, float(reward), next_state, done))
            state = next_state
            logger.debug(f"run_episode: step {steps} complete, total_reward={total_reward:.3f}, solved={solved}, done={done}")

        planner_used = bool(planner_plan_found and not planner_fallback_to_rl and planner_action_steps > 0)
        planner_solved = bool(planner_used and solved)
        logger.info(
            f"run_episode: step loop ended after {steps} steps in {time.perf_counter() - episode_start:.3f}s, "
            f"solved={solved}, planner_used={planner_used}, planner_solved={planner_solved}, "
            f"planner_action_steps={planner_action_steps}, rl_action_steps={rl_action_steps}, "
            f"planner_fallback_to_rl={planner_fallback_to_rl}"
        )

        logger.info(f"run_episode: appending trajectory ({len(trajectory)} transitions) to trajectories list")
        self.trajectories.append(trajectory)

        logger.info(f"run_episode: fitting AML with {len(self.trajectories)} total trajectory(ies)")
        aml_start = time.perf_counter()
        self.aml.fit(self.trajectories)
        aml_elapsed = time.perf_counter() - aml_start
        logger.info(f"run_episode: AML fit completed in {aml_elapsed:.3f}s")

        if hasattr(self.rl_agent, "observe_trajectory"):
            logger.info(f"run_episode: calling rl_agent.observe_trajectory()")
            obs_start = time.perf_counter()
            self.rl_agent.observe_trajectory(trajectory)
            obs_elapsed = time.perf_counter() - obs_start
            logger.info(f"run_episode: observe_trajectory() completed in {obs_elapsed:.3f}s")

        if train_rl:
            logger.info(f"run_episode: starting RL training batch (rl_train_steps={rl_train_steps}, agent type={type(self.rl_agent).__name__})")
            learn_start = time.perf_counter()
            logger.info(f"run_episode: calling rl_agent.learn(total_timesteps={rl_train_steps}, reset_num_timesteps=False)")
            self.rl_agent.learn(total_timesteps=rl_train_steps, reset_num_timesteps=False)
            learn_elapsed = time.perf_counter() - learn_start
            logger.info(f"run_episode: RL training batch completed in {learn_elapsed:.3f}s")
            latest_rl_loss = self._get_latest_rl_loss()
            logger.info(f"run_episode: latest RL loss={latest_rl_loss}")
        else:
            logger.info("run_episode: skipping RL training (train_rl=False)")
            learn_elapsed = 0.0
            latest_rl_loss = float("nan")

        total_elapsed = time.perf_counter() - episode_start
        logger.info(f"run_episode: complete, solved={solved}, steps={steps}, total_reward={total_reward:.3f}, elapsed={total_elapsed:.3f}s")
        return {
            "solved": solved,
            "steps": steps,
            "reward": total_reward,
            "planner_used": planner_used,
            "planner_plan_found": planner_plan_found,
            "planner_fallback_to_rl": planner_fallback_to_rl,
            "planner_solved": planner_solved,
            "planner_backend": getattr(self.planner, "last_backend", "unknown"),
            "planner_error": planner_error,
            "rl_loss": latest_rl_loss,
            "rl_batch_seconds": learn_elapsed,
            "episode_duration_seconds": total_elapsed,
        }

    def train(
        self,
        episodes: int = 100,
        rl_train_steps: int = 250,
        max_total_seconds: Optional[float] = None,
        max_episode_seconds: Optional[float] = None,
        show_progress: bool = False,
    ) -> RampMetrics:
        logger.info(f"Training: starting {episodes} episodes (max_total_seconds={max_total_seconds}, max_episode_seconds={max_episode_seconds})")
        successes: List[int] = []
        success_rate_ma25: List[float] = []
        cumulative_solution_length: List[int] = []
        solved_steps_acc = 0
        efficiency_episode_90pct: Optional[int] = None
        planner_backends: List[str] = []
        rl_backend = str(getattr(self.rl_agent, "_ramp_backend", type(self.rl_agent).__name__))
        terminated_early = False
        timeout_reason: Optional[str] = None
        train_start = time.perf_counter()
        episode_rewards: List[float] = []
        episode_steps: List[int] = []
        episode_successes: List[int] = []
        episode_planner_used: List[int] = []
        episode_planner_plan_found: List[int] = []
        episode_planner_fallback_to_rl: List[int] = []
        episode_planner_solved: List[int] = []
        episode_planner_errors: List[str] = []
        episode_rl_losses: List[float] = []
        episode_rl_batch_seconds: List[float] = []
        episode_durations_seconds: List[float] = []

        episode_iterator = tqdm(range(episodes), disable=not show_progress, desc="RAMP episodes", unit="ep")
        for ep in episode_iterator:
            logger.info(f"Episode {ep}: starting")
            elapsed_total = time.perf_counter() - train_start
            if max_total_seconds is not None and elapsed_total > max_total_seconds:
                terminated_early = True
                timeout_reason = f"Training exceeded time limit of {max_total_seconds:.1f}s"
                logger.warning(timeout_reason)
                break

            try:
                outcome = self.run_episode(
                    train_rl=True,
                    rl_train_steps=rl_train_steps,
                    max_episode_seconds=max_episode_seconds,
                )
            except TimeoutError as exc:
                terminated_early = True
                timeout_reason = str(exc)
                break
            planner_backends.append(str(outcome.get("planner_backend", "unknown")))
            is_success = 1 if outcome["solved"] else 0
            successes.append(is_success)
            episode_successes.append(is_success)
            episode_rewards.append(float(outcome["reward"]))
            episode_steps.append(int(outcome["steps"]))
            episode_planner_used.append(1 if outcome.get("planner_used") else 0)
            episode_planner_plan_found.append(1 if outcome.get("planner_plan_found") else 0)
            episode_planner_fallback_to_rl.append(1 if outcome.get("planner_fallback_to_rl") else 0)
            episode_planner_solved.append(1 if outcome.get("planner_solved") else 0)
            episode_planner_errors.append(str(outcome.get("planner_error") or ""))
            episode_rl_losses.append(float(outcome.get("rl_loss", float("nan"))))
            episode_rl_batch_seconds.append(float(outcome.get("rl_batch_seconds", 0.0)))
            episode_durations_seconds.append(float(outcome.get("episode_duration_seconds", 0.0)))
            if is_success:
                solved_steps_acc += int(outcome["steps"])

            ma = self._moving_average_success(successes, window=25)
            success_rate_ma25.append(ma)
            cumulative_solution_length.append(solved_steps_acc)

            if efficiency_episode_90pct is None and len(successes) >= 25 and ma >= 0.9:
                efficiency_episode_90pct = ep + 1

            if show_progress:
                episode_iterator.set_postfix(
                    success_ma25=f"{ma:.2f}",
                    elapsed_s=f"{(time.perf_counter() - train_start):.1f}",
                )
            logger.info(f"Episode {ep}: complete (solved={is_success}, steps={outcome.get('steps', 0)}, elapsed={time.perf_counter() - train_start:.1f}s)")

        logger.info(f"Training: complete (terminated_early={terminated_early}, total_time={time.perf_counter() - train_start:.1f}s)")
        return RampMetrics(
            success_rate_ma25=success_rate_ma25,
            cumulative_solution_length=cumulative_solution_length,
            efficiency_episode_90pct=efficiency_episode_90pct,
            planner_backend_counts=dict(Counter(planner_backends)),
            rl_backend=rl_backend,
            terminated_early=terminated_early,
            timeout_reason=timeout_reason,
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_successes=episode_successes,
            episode_planner_used=episode_planner_used,
            episode_planner_plan_found=episode_planner_plan_found,
            episode_planner_fallback_to_rl=episode_planner_fallback_to_rl,
            episode_planner_solved=episode_planner_solved,
            episode_planner_backends=planner_backends,
            episode_planner_errors=episode_planner_errors,
            episode_rl_losses=episode_rl_losses,
            episode_rl_batch_seconds=episode_rl_batch_seconds,
            episode_durations_seconds=episode_durations_seconds,
        )
