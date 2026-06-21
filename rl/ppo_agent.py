from stable_baselines3 import PPO


def build_ppo(env, learning_rate=3e-4, gamma=0.999, verbose=0):
    model = PPO(
        policy="MultiInputPolicy" if hasattr(env.observation_space, "spaces") else "MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=64,
        batch_size=64,
        verbose=verbose,
    )
    setattr(model, "_ramp_backend", "ppo")
    return model

