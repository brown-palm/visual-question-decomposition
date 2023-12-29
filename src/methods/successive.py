from dataclasses import dataclass
from enum import Enum

from PIL import Image

from src.methods import VQAModel, VQAModelResult
from src.methods.prompts.successive import (
    SuccessiveExample,
    vqav2_icl_examples,
    gqa_icl_examples,
    okvqa_icl_examples,
    aokvqa_icl_examples,
    scienceqa_icl_examples,
)
from src.models import TextGenerationModel, load_model


@dataclass
class SuccessiveModelResult(VQAModelResult):
    sub_questions: list[str]
    sub_answers: list[str]


class SuccessiveVQAModel(VQAModel):
    def __init__(self, lm_type: str, vlm_type: str, dataset_name: str):
        self.lm: TextGenerationModel = load_model(lm_type)
        self.vlm: TextGenerationModel = load_model(vlm_type)
        self.dataset_name = dataset_name

    def _run(self, question: str, choices: list[str], image: Image) -> VQAModelResult:
        multiple_choice = choices is not None

        if self.dataset_name == "vqav2":
            icl_examples = vqav2_icl_examples()
        elif self.dataset_name == "gqa":
            icl_examples = gqa_icl_examples()
        elif self.dataset_name == "okvqa":
            icl_examples = okvqa_icl_examples()
        elif self.dataset_name == "aokvqa":
            icl_examples = aokvqa_icl_examples(use_choices=multiple_choice)
        elif self.dataset_name == "scienceqa":
            icl_examples = scienceqa_icl_examples()

        # Build test example

        caption = self.vlm.generate_text(image=image)

        test_example = SuccessiveExample(
            question=question,
            choices=choices,
            sub_questions=["What is in this image?"],
            sub_answers=[caption],
        )

        # LM & VLM Iterations

        class PromptingState(Enum):
            QUERY_FOR_LM = 1
            QUERY_FOR_VLM = 2
            FINAL = 3

        def successive_prompt(examples: list[SuccessiveExample]) -> str:
            return "\n\n".join(
                [
                    'Please answer visual questions by asking relevant follow-up questions, one at a time. Follow-up questions will be answered by a vision-language model. Only ask as many follow-up questions as needed to be certain, then provide an "Answer to the original question". All answers should be expressed as short phrases.'
                ]
                + [str(e) for e in examples]
            )

        def vqa_prompt(question):
            return (
                f"Question: {question} " "Short answer:"
                if self.vlm.model_name.startswith("blip2-flan-t5")
                else "Answer:"
            )

        s = PromptingState.QUERY_FOR_LM

        while True:
            prompt = successive_prompt(icl_examples + [test_example])

            if s == PromptingState.QUERY_FOR_LM:
                p_followup = self.lm.get_continuation_likelihood(
                    prompt + "\n", SuccessiveExample.followup_prompt, None
                )
                p_final = self.lm.get_continuation_likelihood(
                    prompt + "\n", SuccessiveExample.answer_prompt, None
                )

                if p_final > p_followup or len(test_example.sub_questions) >= 10:
                    s = PromptingState.FINAL
                else:
                    sub_q = self.lm.generate_text(
                        input_text=(prompt + "\n" + SuccessiveExample.followup_prompt),
                        max_new_tokens=20,
                    ).strip()
                    test_example.sub_questions.append(sub_q)
                    s = PromptingState.QUERY_FOR_VLM

            elif s == PromptingState.QUERY_FOR_VLM:
                sub_a = self.vlm.generate_text(
                    input_text=vqa_prompt(test_example.sub_questions[-1]),
                    image=image,
                    max_new_tokens=20,
                )
                test_example.sub_answers.append(sub_a)
                s = PromptingState.QUERY_FOR_LM

            elif s == PromptingState.FINAL:
                if multiple_choice:
                    probs = [
                        self.lm.get_continuation_likelihood(
                            prompt + SuccessiveExample.answer_prompt, f" {c}", None
                        )
                        for c in choices
                    ]
                    argmax = max(range(len(probs)), key=lambda i: probs[i])
                    test_example.answer = choices[argmax]
                else:
                    test_example.answer = self.lm.generate_text(
                        input_text=(prompt + SuccessiveExample.answer_prompt), max_new_tokens=20
                    ).strip()
                break

        return SuccessiveModelResult(
            prediction=test_example.answer,
            sub_questions=test_example.sub_questions,
            sub_answers=test_example.sub_answers,
        )

    def direct_answer(self, question: str, image: Image) -> VQAModelResult:
        return self._run(question, None, image)

    def multiple_choice(self, question: str, choices: list[str], image: Image) -> VQAModelResult:
        return self._run(question, choices, image)
