from copy import deepcopy
from dataclasses import dataclass

from rich.progress import track

from src.data import QuestionId, VQADataset
from src.methods import VQAModelResult
from src.models import GPT


@dataclass
class Accuracy:
    scores: dict[QuestionId, float]
    acc: float


def exact_match_accuracy(
    dataset: VQADataset, results: dict[QuestionId, VQAModelResult]
) -> Accuracy:
    scores = {}
    for q_id, vqa_item in dataset["entries"].items():
        prediction = results[q_id].prediction if q_id in results else ""
        scores[q_id] = float(prediction == vqa_item["answer"])
    acc = 100 * sum(scores.values()) / len(scores)
    return Accuracy(scores=scores, acc=acc)


def postprocess_choices(dataset: VQADataset, results: dict[QuestionId, VQAModelResult], gpt_eval_model: str = "text-davinci-003") -> dict:
    results = deepcopy(results)
    eval_model = GPT(gpt_eval_model)

    for q_id, model_result in track(results.items()):
        pred = model_result.prediction or ""
        if pred == "":
            continue

        choices = dataset["entries"][q_id]["choices"]
        if pred not in choices:
            prompt = (
                f"Choices: {', '.join(choices)}\n" f"Candidate: {pred}\n" f"Most similar choice:"
            )
            probs = [eval_model.get_continuation_likelihood(prompt, f" {c}") for c in choices]
            argmax = max(range(len(probs)), key=lambda i: probs[i])
            model_result.prediction = choices[argmax]

    return dict(dataset=dataset, results=results)


from .aokvqa import aokvqa_da_accuracy  # noqa: E402
from .kamalloo import kamalloo_2023_accuracy  # noqa: E402
from .vqa import vqa_accuracy  # noqa: E402

__all__ = [
    "Accuracy",
    "exact_match_accuracy",
    "postprocess_choices",
    "aokvqa_da_accuracy",
    "kamalloo_2023_accuracy",
    "vqa_accuracy",
]
