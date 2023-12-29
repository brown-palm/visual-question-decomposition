import os
import random

from tango import step

from src.data import (
    VQADataset,
    load_aokvqa,
    load_gqa,
    load_okvqa,
    load_scienceqa,
    load_vqav2,
)
from src.eval import (
    Accuracy,
    aokvqa_da_accuracy,
    exact_match_accuracy,
    kamalloo_2023_accuracy,
    postprocess_choices,
    vqa_accuracy,
)
from src.methods import End2EndVQAModel, SuccessiveVQAModel, ViperVQAModel
from src.predict import VQADatasetResults, inference_loop


@step(deterministic=True, cacheable=False, version="001")
def load_dataset(dataset_name: str, split: str, n: int = None) -> VQADataset:
    if dataset_name == "vqav2":
        dataset = load_vqav2(split, os.environ["VQA_DIR"], os.environ["COCO_DIR"])
    elif dataset_name == "gqa":
        dataset = load_gqa(split, "balanced", None, os.environ["GQA_DIR"])
    elif dataset_name == "okvqa":
        dataset = load_okvqa(split, os.environ["OKVQA_DIR"], os.environ["COCO_DIR"])
    elif dataset_name == "aokvqa":
        dataset = load_aokvqa(split, os.environ["AOKVQA_DIR"], os.environ["COCO_DIR"])
    elif dataset_name == "scienceqa":
        dataset = load_scienceqa(split)

    if n is not None:
        random.seed(1)
        keep_keys = random.sample(list(dataset["entries"].keys()), n)
        dataset["entries"] = {k: v for k, v in dataset["entries"].items() if k in keep_keys}

    return dataset


@step(bind=True, deterministic=True, cacheable=True, version="001")
def run_inference(
    self, method: str, method_kwargs: dict[str], dataset: VQADataset
) -> VQADatasetResults:
    if method == "e2e":
        model_class = End2EndVQAModel
    elif method == "viper":
        model_class = ViperVQAModel
    elif method == "successive":
        model_class = SuccessiveVQAModel
        method_kwargs["dataset_name"] = dataset["dataset_name"]

    vqa_model = model_class(**method_kwargs)
    results = inference_loop(vqa_model, dataset, cache_dir=self.work_dir_for_run)
    return results


@step(deterministic=True, cacheable=True, version="001")
def evaluate(
    dataset_name: str, dataset: VQADataset, dataset_results: VQADatasetResults, gpt_eval_model: str
) -> dict[str, Accuracy]:
    if dataset_name == "vqav2":
        return dict(
            instructgpt_acc=kamalloo_2023_accuracy(
                dataset=dataset,
                results=dataset_results["da_results"],
                use_answer_list=True,
                gpt_eval_model=gpt_eval_model,
            ),
            vqav2_acc=vqa_accuracy(dataset=dataset, results=dataset_results["da_results"]),
        )
    elif dataset_name == "gqa":
        return dict(
            instructgpt_acc=kamalloo_2023_accuracy(
                dataset=dataset,
                results=dataset_results["da_results"],
                use_answer_list=False,
                gpt_eval_model=gpt_eval_model,
            ),
            gqa_acc=exact_match_accuracy(dataset=dataset, results=dataset_results["da_results"]),
        )
    elif dataset_name == "okvqa":
        return dict(
            instructgpt_acc=kamalloo_2023_accuracy(
                dataset=dataset,
                results=dataset_results["da_results"],
                use_answer_list=True,
                gpt_eval_model=gpt_eval_model,
            ),
            okvqa_acc=vqa_accuracy(dataset=dataset, results=dataset_results["da_results"]),
        )
    elif dataset_name == "aokvqa":
        return dict(
            instructgpt_acc=kamalloo_2023_accuracy(
                dataset=dataset,
                results=dataset_results["da_results"],
                use_answer_list=True,
                gpt_eval_model=gpt_eval_model,
            ),
            aokvqa_da_acc=aokvqa_da_accuracy(
                dataset=dataset, results=dataset_results["da_results"]
            ),
            aokvqa_mc_acc=exact_match_accuracy(
                **postprocess_choices(
                    dataset=dataset,
                    results=dataset_results["mc_results"],
                    gpt_eval_model=gpt_eval_model,
                )
            ),
        )
    elif dataset_name == "scienceqa":
        return dict(
            scienceqa_acc=exact_match_accuracy(
                **postprocess_choices(
                    dataset=dataset,
                    results=dataset_results["mc_results"],
                    gpt_eval_model=gpt_eval_model,
                )
            ),
        )


@step(deterministic=True, cacheable=True, version="001")
def process_metrics(metrics: dict[str, Accuracy]) -> dict[str, float]:
    return {k: v.acc for k, v in metrics.items()}
