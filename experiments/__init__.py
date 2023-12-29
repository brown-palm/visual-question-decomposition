from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tango import Step, StepGraph
from tango.cli import execute_step_graph, prepare_workspace, tango_cli


@dataclass
class Experiment(ABC):
    @property
    @abstractmethod
    def step_dict(self) -> dict[str, Step]:
        raise NotImplementedError

    @property
    def step_graph(self) -> StepGraph:
        return StepGraph(self.step_dict)

    def run(self):
        with tango_cli():
            execute_step_graph(step_graph=self.step_graph)

    def result(self, step_name: str) -> Any:
        ws = prepare_workspace()
        return self.step_dict[step_name].result(workspace=ws)


__all__ = [
    "Experiment",
]
