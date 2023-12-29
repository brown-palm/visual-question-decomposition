import json

from rich.progress import track

from . import VQADataset, get_coco_path


def load_aokvqa(split: str, aokvqa_dir: str, coco_dir: str) -> VQADataset:
    assert split in ["train", "val", "test"]

    entries = {}

    aokvqa_data = json.load(open(f"{aokvqa_dir}/aokvqa_v1p0_{split}.json"))

    for q in track(aokvqa_data):
        q_id = str(q["question_id"])
        entries[q_id] = {
            "question": q["question"],
            "choices": q["choices"],
            "image": get_coco_path(coco_dir, f"{split}2017", q["image_id"]),
            "answer": q["choices"][q["correct_choice_idx"]],
            "answer_list": (q["direct_answers"] if q["difficult_direct_answer"] is False else None),
        }

    return VQADataset(
        dataset_name="aokvqa", direct_answer=True, multiple_choice=True, entries=entries
    )
