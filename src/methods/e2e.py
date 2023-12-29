from PIL import Image

from src.methods import VQAModel, VQAModelResult
from src.models import TextGenerationModel, load_model


class End2EndVQAModel(VQAModel):
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model: TextGenerationModel = load_model(model_type)

    def _get_prompt(self, question: str) -> str:
        if "flan-t5" in self.model_type:
            return f"Question: {question} Short answer:"
        elif "opt" in self.model_type:
            return f"Question: {question} Answer:"

    def _get_answer_text(self, answer: str) -> str:
        return f" {answer}"

    def direct_answer(self, question: str, image: Image) -> VQAModelResult:
        prompt = self._get_prompt(question)
        answer = self.model.generate_text(input_text=prompt, image=image, max_new_tokens=20)
        return VQAModelResult(prediction=answer)

    def multiple_choice(self, question: str, choices: list[str], image: Image) -> VQAModelResult:
        prompt = self._get_prompt(question)
        continuations = [self._get_answer_text(c) for c in choices]
        probs = [self.model.get_continuation_likelihood(prompt, c, image) for c in continuations]
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        return VQAModelResult(prediction=choices[argmax])
