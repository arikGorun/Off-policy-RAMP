from collections import defaultdict
import numpy as np


class NumericActionModelLearner:
    def __init__(self):
        self.model = {}

    @staticmethod
    def _to_vector(state):
        if isinstance(state, dict):
            parts = []
            for key in sorted(state.keys()):
                value = state[key]
                parts.append(np.asarray(value, dtype=np.float32).ravel())
            return np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)

        return np.asarray(state, dtype=np.float32).ravel()

    def fit(self, trajectories):
        grouped = defaultdict(list)

        for traj in trajectories:
            for transition in traj:
                if len(transition) == 4:
                    s, a, _, sp = transition
                else:
                    s, a, _, sp, _ = transition

                grouped[int(a)].append((self._to_vector(s), self._to_vector(sp)))

        learned = {}

        for action, transitions in grouped.items():
            deltas = [sp - s for s,sp in transitions]
            learned[action] = {
                "mean_effect": np.mean(deltas, axis=0),
                "count": len(deltas),
            }

        self.model = learned
        return learned
