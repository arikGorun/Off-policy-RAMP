from stable_baselines3 import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def build_her(env, learning_rate=1e-3, gamma=0.999, verbose=1):

    is_goal_env = hasattr(env.observation_space, "spaces") and {
        "observation",
        "achieved_goal",
        "desired_goal",
    }.issubset(set(env.observation_space.spaces.keys()))

    if not is_goal_env:
        raise RuntimeError(
            "HER requires a goal-conditioned environment exposing observation/achieved_goal/desired_goal keys."
        )

    model = DQN(
        "MultiInputPolicy" if hasattr(env.observation_space, "spaces") else "MlpPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=verbose,
    )
    setattr(model, "_ramp_backend", "her_native")
    return model


