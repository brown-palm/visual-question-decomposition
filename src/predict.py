from typing import TypedDict

from PIL import Image
from rich.progress import track
from sqlitedict import SqliteDict

from src.data import QuestionId, VQADataset
from src.methods import VQAModel, VQAModelResult


class VQADatasetResults(TypedDict):
    da_results: dict[QuestionId, VQAModelResult]
    mc_results: dict[QuestionId, VQAModelResult]


def inference_loop(
    vqa_model: VQAModel, dataset: VQADataset, cache_dir: str = None
) -> VQADatasetResults:
    da_results: dict[QuestionId, VQAModelResult]
    mc_results: dict[QuestionId, VQAModelResult]

    if cache_dir is not None:
        da_results = SqliteDict(f"{cache_dir}/da_results.sqlite", "sparse_sequence", flag="c")
        mc_results = SqliteDict(f"{cache_dir}/mc_results.sqlite", "sparse_sequence", flag="c")
    else:
        da_results = {}
        mc_results = {}

    for q_id, vqa_item in track(dataset["entries"].items()):
        if q_id in da_results or q_id in mc_results:
            continue

        question = vqa_item["question"]
        choices = vqa_item["choices"]
        if isinstance(vqa_item["image"], Image.Image):
            image = vqa_item["image"]
        elif isinstance(vqa_item["image"], str):
            image = Image.open(vqa_item["image"]).convert("RGB")

        if dataset["direct_answer"]:
            da_results[q_id] = vqa_model.direct_answer(question, image)
        if dataset["multiple_choice"]:
            mc_results[q_id] = vqa_model.multiple_choice(question, choices, image)

        if cache_dir is not None:
            da_results.commit()
            mc_results.commit()

    if cache_dir is not None:
        da_results = dict(da_results)
        mc_results = dict(mc_results)

    return VQADatasetResults(da_results=(da_results or None), mc_results=(mc_results or None))
