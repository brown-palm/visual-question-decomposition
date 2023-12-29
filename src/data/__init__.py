from typing import Optional, TypedDict, Union

from PIL import Image

QuestionId = str


class VQAItem(TypedDict):
    question: str
    choices: Optional[list[str]]
    image: Union[str, Image.Image]
    answer: Optional[str]
    answer_list: Optional[list[str]]
    question_type: Optional[str]


class VQADataset(TypedDict):
    dataset_name: str
    direct_answer: bool
    multiple_choice: bool
    entries: dict[QuestionId, VQAItem]


def get_coco_path(coco_dir: str, split: str, image_id: int) -> str:
    if split.endswith("2017"):
        return f"{coco_dir}/{split}/{image_id:012}.jpg"
    else:
        return f"{coco_dir}/{split}/COCO_{split}_{image_id:012}.jpg"


from .aokvqa import load_aokvqa  # noqa: E402
from .gqa import load_gqa  # noqa: E402
from .okvqa import load_okvqa  # noqa: E402
from .scienceqa import load_scienceqa  # noqa: E402
from .vqav2 import load_vqav2  # noqa: E402

__all__ = [
    "QuestionId",
    "VQAItem",
    "VQADataset",
    "get_coco_path",
    "load_vqav2",
    "load_gqa",
    "load_okvqa",
    "load_aokvqa",
    "load_scienceqa",
]
