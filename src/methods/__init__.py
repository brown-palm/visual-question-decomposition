from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class VQAModelResult:
    prediction: str


class VQAModel(ABC):
    @abstractmethod
    def direct_answer(self, question: str, image: Image) -> VQAModelResult:
        pass

    @abstractmethod
    def multiple_choice(self, question: str, choices: List[str], image: Image) -> VQAModelResult:
        pass


from .e2e import End2EndVQAModel  # noqa: E402
from .successive import SuccessiveModelResult, SuccessiveVQAModel  # noqa: E402
from .viper import ViperModelResult, ViperVQAModel  # noqa: E402

__all__ = [
    "VQAModelResult",
    "VQAModel",
    "End2EndVQAModel",
    "SuccessiveModelResult",
    "SuccessiveVQAModel",
    "ViperModelResult",
    "ViperVQAModel",
]
