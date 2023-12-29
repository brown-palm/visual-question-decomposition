import json

from rich.progress import track

from . import VQADataset, get_coco_path


def load_okvqa(split: str, okvqa_dir: str, coco_dir: str) -> VQADataset:
    assert split in ["train2014", "val2014"]

    entries = {}

    okvqa_questions = json.load(open(f"{okvqa_dir}/OpenEnded_mscoco_{split}_questions.json", "r"))[
        "questions"
    ]

    okvqa_annotations = json.load(open(f"{okvqa_dir}/mscoco_{split}_annotations.json", "r"))[
        "annotations"
    ]
    answer_lists = [[x["answer"] for x in x["answers"]] for x in okvqa_annotations]
    question_types = [x["question_type"] for x in okvqa_annotations]

    for q, answer_list, q_type in track(zip(okvqa_questions, answer_lists, question_types)):
        q_id = str(q["question_id"])
        entries[q_id] = {
            "question": q["question"],
            "choices": None,
            "image": get_coco_path(coco_dir, split, q["image_id"]),
            "answer": None,
            "answer_list": answer_list,
            "question_type": q_type,
        }

    return VQADataset(
        dataset_name="okvqa", direct_answer=True, multiple_choice=False, entries=entries
    )
