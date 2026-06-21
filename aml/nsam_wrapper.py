from pathlib import Path


class NSAMWrapper:
    def __init__(self, domain_path: Path | None = None):
        self.domain_path = domain_path
        self.model = {}

    def fit(self, trajectories):
        raise NotImplementedError(
            "NSAMWrapper has no fallback path. Implement numeric-sam Observation conversion before using it in strict mode."
        )
