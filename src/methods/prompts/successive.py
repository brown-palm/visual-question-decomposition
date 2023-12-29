from typing import ClassVar
from dataclasses import dataclass, field
from itertools import zip_longest


@dataclass
class SuccessiveExample:
    question: list[str]
    choices: list[str] = None
    answer: str = None
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[str] = field(default_factory=list)

    question_prompt: ClassVar[str] = "Question:"
    choices_prompt: ClassVar[str] = "Choices:"
    followup_prompt: ClassVar[str] = "Follow-up:"
    followup_answer_prompt: ClassVar[str] = "Follow-up answer:"
    answer_prompt: ClassVar[str] = "Answer to the original question:"

    def __str__(self) -> str:
        prompt = []

        prompt.append(f"{self.question_prompt} {self.question}")
        if self.choices is not None:
            prompt.append(f"{self.choices_prompt} {', '.join(self.choices)}")

        for sub_q, sub_a in zip_longest(self.sub_questions, self.sub_answers, fillvalue=None):
            prompt.append(f"{self.followup_prompt} {sub_q}")
            if sub_a is not None:
                prompt.append(f"{self.followup_answer_prompt} {sub_a}")

        if self.answer is not None:
            prompt.append(f"{self.answer_prompt} {self.answer}")

        return "\n".join(prompt)


def vqav2_icl_examples() -> list[SuccessiveExample]:
    return [
        # 393762003
        SuccessiveExample(
            question="What is covering the ground?",
            choices=None,
            answer="grass",
            sub_questions=["What is in this image?", "What is covering the ground in this field?"],
            sub_answers=["a sign post in the middle of a field", "grass"],
        ),
        # 267311002
        SuccessiveExample(
            question="What is the lady wearing on her face?",
            choices=None,
            answer="sunglasses",
            sub_questions=["What is in this image?", "What is the woman wearing on her face?"],
            sub_answers=["a woman is playing tennis on a court", "sunglasses"],
        ),
        # 145841001
        SuccessiveExample(
            question="Does this computer have external speakers?",
            choices=None,
            answer="yes",
            sub_questions=["What is in this image?", "Are there any speakers on the desk?"],
            sub_answers=["a computer desk with a monitor, keyboard and mouse", "yes"],
        ),
    ]


def gqa_icl_examples() -> list[SuccessiveExample]:
    return [
        # 0876219
        SuccessiveExample(
            question="Are there both windows and doors in this photograph?",
            choices=None,
            answer="yes",
            sub_questions=[
                "What is in this image?",
                "Does the small blue train have windows?",
                "Does the small blue train have doors?",
            ],
            sub_answers=[
                "a small blue train",
                "yes",
                "yes",
            ],
        ),
        # 12989761
        SuccessiveExample(
            question="Is the woman to the right of a kite?",
            choices=None,
            answer="no",
            sub_questions=[
                "What is in this image?",
                "Is the woman to the right of a kite?",
            ],
            sub_answers=[
                "a woman flying a kite in the sky",
                "no",
            ],
        ),
        # 19125192
        SuccessiveExample(
            question="Are there any balloons on the pole?",
            choices=None,
            answer="no",
            sub_questions=[
                "What is in this image?",
                "Where is the pole?",
                "What is on the pole on top of the building?",
                "Are there any balloons on the pole on top of the building?",
            ],
            sub_answers=[
                "a large building with a clock tower on top",
                "the top of the building",
                "flag",
                "no",
            ],
        ),
    ]


def okvqa_icl_examples() -> list[SuccessiveExample]:
    return [
        # 192515
        SuccessiveExample(
            question="How do you care for this equipment?",
            choices=None,
            answer="clean",
            sub_questions=[
                "What is in this image?",
                "How do you care for this surfboard?",
            ],
            sub_answers=[
                "a man with a surfboard walking into the ocean",
                "Clean it with a soft cloth and a mild soap",
            ],
        ),
        # 4746145
        SuccessiveExample(
            question="What is the menu at this restaurant?",
            choices=None,
            answer="korean bbq taco",
            sub_questions=[
                "What is in this image?",
                "What is written on the food truck?",
            ],
            sub_answers=["a food truck is parked on the street", "korean bbq taco"],
        ),
        # 314345
        SuccessiveExample(
            question="What is the title commonly given to the man wearing the red tie and green vest?",
            choices=None,
            answer="bartender",
            sub_questions=[
                "What is in this image?",
                "Who is the man in the red tie and green vest?",
            ],
            sub_answers=["a group of people sitting at a bar", "a bartender"],
        ),
    ]


def aokvqa_icl_examples(use_choices: bool = False) -> list[SuccessiveExample]:
    return [
        # 3KvSxMXMnkwcR7umqZj9n2
        SuccessiveExample(
            question="What type of surfboard is the man in green holding?",
            choices=["midboard", "bodyboard", "shortboard", "longboard"] if use_choices else None,
            answer="longboard",
            sub_questions=["What is in this image?", "How long is the man in green's surfboard?"],
            sub_answers=[
                "a group of people are walking into the ocean with surfboards",
                "ten feet",
            ],
        ),
        # 5JFinSUU5VbsNwLAdPQ82v
        SuccessiveExample(
            question="What does the vehicle look like?",
            choices=["motorcycle", "tank", "boat", "car"] if use_choices else None,
            answer="motorcycle",
            sub_questions=[
                "What is in this image?",
            ],
            sub_answers=[
                "a red motorcycle with a sidecar",
            ],
        ),
        # 24vcZcaxamEJwTD7wHYb4v
        SuccessiveExample(
            question="In which country is this bus located?",
            choices=["uk", "cuba", "usa", "mexico"] if use_choices else None,
            answer="uk",
            sub_questions=[
                "What is in this image?",
                "What city is known for red double-decker buses?",
                "What country is London in?",
            ],
            sub_answers=[
                "a red double decker bus",
                "London",
                "United Kingdom",
            ],
        ),
    ]


def scienceqa_icl_examples() -> list[SuccessiveExample]:
    return [
        # 381
        SuccessiveExample(
            question="Based on the arrows, which of the following organisms is an omnivore?",
            choices=["barren-ground caribou", "grizzly bear"],
            answer="grizzly bear",
            sub_questions=[
                "What is in this image?",
                "What organisms are in this image?",
                "Is the grizzly bear an omniovore?",
                "Is the barren-ground caribou an omniovore?",
            ],
            sub_answers=[
                "the life cycle of a bear",
                "a bear, a fox, a wolf, a moose",
                "yes",
                "no",
            ],
        ),
        # 3249
        SuccessiveExample(
            question="What can Gordon and Roxanne trade to each get what they want?",
            choices=[
                "Gordon can trade his tomatoes for Roxanne's sandwich.",
                "Gordon can trade his tomatoes for Roxanne's broccoli.",
                "Roxanne can trade her almonds for Gordon's tomatoes.",
                "Roxanne can trade her broccoli for Gordon's oranges.",
            ],
            answer="Gordon can trade his tomatoes for Roxanne's broccoli.",
            sub_questions=[
                "What is in this image?",
                "What is in Gordon's lunch box?",
                "What is in Roxanne's lunch box?",
                "Does Gordon have tomatoes or oranges?",
                "Does Roxanne have a sandwich, broccoli, or almonds?",
            ],
            sub_answers=[
                "two images of a lunch box with different foods",
                "a sandwich, a bottle of water, and a piece of fruit",
                "a sandwich, a fruit, a vegetable, a yogurt, a water bottle",
                "tomatoes",
                "sandwich and almonds",
            ],
        ),
        # 4930
        SuccessiveExample(
            question="Which bird's beak is also adapted to get nectar out of long flowers?",
            choices=["bufflehead", "bronzy sunbird"],
            answer="bronzy sunbird",
            sub_questions=[
                "What is in this image?",
                "What is the beak doing?",
            ],
            sub_answers=["a hummingbird with a red beak and green body", "piercing the flower"],
        ),
    ]
