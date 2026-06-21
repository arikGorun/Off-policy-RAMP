import random
from collections import deque


class PlannerReplayBuffer:
    def __init__(self, capacity=100_000, planner_boost=2.0):
        self.capacity = int(capacity)
        self.planner_boost = float(planner_boost)
        self._storage = deque(maxlen=self.capacity)

    def add(self, transition, from_planner=False):
        weight = self.planner_boost if from_planner else 1.0
        self._storage.append((transition, weight))

    def extend_trajectory(self, trajectory, from_planner=False):
        for transition in trajectory:
            self.add(transition, from_planner=from_planner)

    def sample(self, batch_size):
        if not self._storage:
            return []

        transitions, weights = zip(*self._storage)
        return random.choices(transitions, weights=weights, k=min(int(batch_size), len(transitions)))

    def __len__(self):
        return len(self._storage)
