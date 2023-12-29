from src.data import QuestionId, VQADataset
from src.methods import VQAModelResult

from . import Accuracy


# https://github.com/allenai/aokvqa/blob/42e8c44ff40d13cf272776a3197365f582137772/evaluation/eval_predictions.py#L9
def aokvqa_da_accuracy(dataset: VQADataset, results: dict[QuestionId, VQAModelResult]) -> Accuracy:
    scores = {}

    for q_id, vqa_item in dataset["entries"].items():
        if vqa_item["answer_list"] is None:
            continue

        if q_id not in results:
            scores[q_id] = 0.0
            continue

        prediction = results[q_id].prediction
        num_match = sum([prediction == ans for ans in vqa_item["answer_list"]])
        scores[q_id] = min(1.0, num_match / 3.0)

    acc = 100 * sum(scores.values()) / len(scores)

    return Accuracy(scores=scores, acc=acc)
