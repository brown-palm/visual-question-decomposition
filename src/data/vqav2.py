import json

from rich.progress import track

from . import VQADataset, get_coco_path


def load_vqav2(split: str, vqa_dir: str, coco_dir: str) -> VQADataset:
    assert split in ["train2014", "val2014", "test-dev2015", "test2015"]

    vqa_questions = json.load(open(f"{vqa_dir}/v2_OpenEnded_mscoco_{split}_questions.json", "r"))[
        "questions"
    ]

    if split in ["train2014", "val2014"]:
        vqa_annotations = json.load(open(f"{vqa_dir}/v2_mscoco_{split}_annotations.json", "r"))[
            "annotations"
        ]
        answer_lists = [[x["answer"] for x in x["answers"]] for x in vqa_annotations]
        question_types = [x["question_type"] for x in vqa_annotations]
    else:
        answer_lists = [None for _ in vqa_questions]
        question_types = [None for _ in vqa_questions]

    coco_split = split if split != "test-dev2015" else "test2015"

    entries = {}

    for q, answer_list, q_type in track(zip(vqa_questions, answer_lists, question_types)):
        q_id = str(q["question_id"])
        entries[q_id] = {
            "question": q["question"],
            "choices": None,
            "image": get_coco_path(coco_dir, coco_split, q["image_id"]),
            "answer": None,
            "answer_list": answer_list,
            "question_type": q_type,
        }

    return VQADataset(
        dataset_name="vqav2", direct_answer=True, multiple_choice=False, entries=entries
    )
