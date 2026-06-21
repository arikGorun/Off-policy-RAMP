from collections import defaultdict
import numpy as np

class NumericActionModelLearner:
    def __init__(self):
        self.model = {}

    def fit(self, trajectories):
        grouped = defaultdict(list)

        for traj in trajectories:
            for s,a,r,sp in traj:
                grouped[a].append((s,sp))

        learned = {}

        for action, transitions in grouped.items():
            deltas = [sp - s for s,sp in transitions]
            learned[action] = {
                "mean_effect": np.mean(deltas, axis=0),
                "count": len(deltas)
            }

        self.model = learned
        return learned
