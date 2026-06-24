from sb3_contrib import QRDQN


def build_rainbow(env, learning_rate=1e-3, gamma=0.999, verbose=1, device="auto"):
    model = QRDQN(
        policy="MultiInputPolicy" if hasattr(env.observation_space, "spaces") else "MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=verbose,
        device=device,
    )
    setattr(model, "_ramp_backend", "qrdqn")
    return model