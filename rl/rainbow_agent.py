from stable_baselines3 import DQN

def build_rainbow(env):
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        gamma=0.999,
        buffer_size=100000,
        batch_size=256,
        verbose=1,
    )
    return model
