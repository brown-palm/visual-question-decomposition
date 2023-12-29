from rich.progress import track

from src.data import QuestionId, VQADataset
from src.methods import VQAModelResult
from src.models import GPT

from . import Accuracy


# Evaluating Open-Domain Question Answering in the Era of Large Language Models (Kamalloo 2023)
def kamalloo_2023_accuracy(
    dataset: VQADataset, results: dict[QuestionId, VQAModelResult], use_answer_list: bool = False, gpt_eval_model: str = "text-davinci-003"
) -> Accuracy:
    eval_model = GPT(gpt_eval_model)

    scores = {}
    for q_id, vqa_item in track(dataset["entries"].items()):
        if use_answer_list and vqa_item["answer_list"] is None:
            continue
        if q_id not in results or results[q_id].prediction in [None, ""]:
            scores[q_id] = 0.0
            continue

        q = vqa_item["question"]
        if use_answer_list:
            a = f"Answers: {' or '.join(vqa_item['answer_list'])}"
        else:
            a = f"Answer: {vqa_item['answer']}"
        c = results[q_id].prediction

        prompt = f"Question: {q}\n" f"{a}\n" f"Candidate: {c}\n\n" f"Is candidate correct?"

        nll_yes = eval_model.get_continuation_likelihood(prompt, " Yes")
        nll_no = eval_model.get_continuation_likelihood(prompt, " No")

        scores[q_id] = float(nll_yes >= nll_no)

    acc = 100 * sum(scores.values()) / len(scores)

    return Accuracy(scores=scores, acc=acc)
