from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    metrics: dict[str, float]


class Experiment:
    name: str

    def prepare(self) -> None:
        ...

    def run(self) -> ExperimentResult:
        ...
