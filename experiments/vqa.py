import dataclasses
from dataclasses import dataclass
from typing import Literal, Optional, Union

import tyro
from rich.console import Console
from rich.table import Table
from tango import Step

# https://github.com/allenai/tango/issues/606
from tango.workspaces.local_workspace import LocalWorkspace  # noqa: F401
from tango.workspaces.memory_workspace import MemoryWorkspace  # noqa: F401

from experiments import Experiment
from experiments.steps import evaluate, load_dataset, process_metrics, run_inference

# Dataset arguments


@dataclass
class dataset_args:
    n: Optional[int] = None
    """ Number of random dataset samples """

    @property
    def name(self):
        return type(self).__name__


@dataclass
class vqav2(dataset_args):
    n: int = 1000
    split: Literal["train2014", "val2014", "test-dev2015", "test2015"] = "val2014"


@dataclass
class gqa(dataset_args):
    split: Literal["testdev", "challenge", "submission", "test", "val", "train"] = "testdev"


@dataclass
class okvqa(dataset_args):
    split: Literal["train2014", "val2014"] = "val2014"


@dataclass
class aokvqa(dataset_args):
    split: Literal["train", "val", "test"] = "val"


@dataclass
class scienceqa(dataset_args):
    split: Literal["train", "validation", "test"] = "validation"


# Method arguments


class method_args:
    @property
    def name(self):
        return type(self).__name__

    @property
    def kwargs(self):
        return dataclasses.asdict(self)


@dataclass
class e2e(method_args):
    model_type: Literal[
        "blip2-opt-2.7b",
        "blip2-opt-6.7b",
        "blip2-flan-t5-xl",
        "blip2-flan-t5-xxl",
    ] = "blip2-flan-t5-xxl"
    """ BLIP-2 model for VQA """


@dataclass
class viper(method_args):
    lm_type: str = "code-davinci-002"
    """ GPT model for code generation """
    prompt_mode: Literal[
        "full_api",
        "no_llm_query",
        "no_simple_query",
        "only_queries",
        "only_queries_vqav2",
        "only_queries_gqa",
        "only_queries_okvqa",
        "only_queries_aokvqa",
        "only_queries_scienceqa",
    ] = "full_api"
    """ API prompt (src/methods/prompts/viper/*.txt) """


@dataclass
class successive(method_args):
    lm_type: str = "text-davinci-002"
    """ GPT model for question decomposition
    (only Completions API models are supported) """
    vlm_type: Literal[
        "blip2-opt-2.7b",
        "blip2-opt-6.7b",
        "blip2-flan-t5-xl",
        "blip2-flan-t5-xxl",
    ] = "blip2-flan-t5-xxl"
    """ BLIP-2 model for answering visual sub-questions """


# Experiment


@dataclass
class VQAExperiment(Experiment):
    dataset: Union[vqav2, gqa, okvqa, aokvqa, scienceqa]
    method: Union[e2e, viper, successive]
    gpt_eval_model: str = "text-davinci-003"
    """ GPT model for LLM-based evaluation and multiple choice matching
        (only Completions API models are supported) """

    @property
    def step_dict(self) -> dict[str, Step]:
        dataset = load_dataset(
            dataset_name=self.dataset.name, split=self.dataset.split, n=self.dataset.n
        )

        results = run_inference(
            method=self.method.name,
            method_kwargs=self.method.kwargs,
            dataset=dataset,
        )

        metrics = evaluate(dataset_name=self.dataset.name, dataset=dataset, dataset_results=results, gpt_eval_model=self.gpt_eval_model)

        metrics_summary = process_metrics(metrics=metrics)

        return {
            "dataset": dataset,
            "results": results,
            "metrics": metrics,
            "metrics_summary": metrics_summary,
        }


def print_metrics(metrics):
    table = Table()
    for k in metrics.keys():
        table.add_column(k)
    table.add_row(*(str(m) for m in metrics.values()))
    Console().print(table)


if __name__ == "__main__":
    experiment = tyro.cli(VQAExperiment)
    experiment.run()
    metrics = experiment.result("metrics_summary")
    print_metrics(metrics)
