from stable_baselines3 import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

def build_her(env):
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        learning_rate=1e-3,
        gamma=0.999,
        verbose=1,
    )
    return model
