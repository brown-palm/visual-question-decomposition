from datasets import load_dataset

from . import VQADataset


def load_scienceqa(split: str) -> VQADataset:
    assert split in ["train", "validation", "test"]

    dataset = load_dataset("derek-thomas/ScienceQA")[split]

    entries = {}
    for q_id, q in enumerate(dataset):
        q_id = str(q_id)

        if q["image"] is None or q["task"] != "closed choice":
            continue

        entries[q_id] = {
            "question": q["question"],
            "choices": q["choices"],
            "image": q["image"],
            "answer": q["choices"][q["answer"]],
            "answer_list": None,
            "question_type": q["topic"],
        }

    return VQADataset(
        dataset_name="scienceqa", direct_answer=False, multiple_choice=True, entries=entries
    )
