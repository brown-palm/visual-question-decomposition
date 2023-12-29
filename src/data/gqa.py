import json
import random
from typing import Optional

from rich.progress import track

from . import VQADataset


def load_gqa(split: str, subset: str, k: Optional[int], gqa_dir: str) -> VQADataset:
    assert split in ["testdev", "challenge", "submission", "test", "val", "train"]
    assert subset in ["all", "balanced"]

    if split == "train" and subset == "all":
        gqa_files = [
            f"{gqa_dir}/questions/train_all_questions/train_all_questions_{i}.json"
            for i in range(10)
        ]
    else:
        gqa_files = [f"{gqa_dir}/questions/{split}_{subset}_questions.json"]

    entries = {}

    for json_fp in gqa_files:
        gqa_data = json.load(open(json_fp, "r"))
        for q_id, q in track(gqa_data.items()):
            q_id = str(q_id)
            entries[q_id] = {
                "question": q["question"],
                "choices": None,
                "image": f"{gqa_dir}/images/{q['imageId']}.jpg",
                "answer": q["answer"],
                "answer_list": None,
                "question_type": f"{q['types']['structural']}{q['types']['semantic'].capitalize()}",
            }

    if k is not None:
        random.seed(1)
        question_types = {}

        for json_fp in gqa_files:
            gqa_data = json.load(open(json_fp, "r"))
            for q_id, q in gqa_data.items():
                q_id = str(q_id)
                q_type = q["types"]["detailed"]
                if q_type not in question_types:
                    question_types[q_type] = []
                question_types[q_type].append(q_id)

        k_subset = []
        for type_ids in question_types.values():
            if len(type_ids) > k:
                k_subset += random.sample(type_ids, k)
            else:
                k_subset += type_ids
        entries = {q_id: entries[q_id] for q_id in k_subset}

    return VQADataset(
        dataset_name="gqa", direct_answer=True, multiple_choice=False, entries=entries
    )
