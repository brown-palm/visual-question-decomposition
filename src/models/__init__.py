from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class TextGenerationModel(ABC):
    model_name: str

    @abstractmethod
    def generate_text(
        self,
        input_text: str,
        image: Image,
        max_new_tokens: int,
        stop: str,
        temperature: float,
    ) -> str:
        pass

    # weighted byte-length likelihood
    def _continuation_wbl_likelihood(
        self, prompt: str, continuation: str, tokens: list[str], logprobs: list[float]
    ):
        token_offsets = [sum(len(t) for t in tokens[:i]) for i in range(len(tokens))]

        total_overlap = 0
        total_likelihood = 0.0

        for i in range(len(tokens)):
            # Calculate the overlap between the token and the continuation
            token_start_pos = token_offsets[i]
            if i + 1 < len(token_offsets):
                token_end_pos = token_offsets[i + 1]
            else:
                token_end_pos = len(prompt + continuation)

            overlap = token_end_pos - max(token_start_pos, len(prompt))

            # If an overlap exists, contribute token logprob to overall likelihood
            if logprobs[i] is not None and overlap > 0:
                total_likelihood += logprobs[i] * overlap
                total_overlap += overlap

        total_likelihood /= total_overlap

        return total_likelihood

    @abstractmethod
    def get_continuation_likelihood(self, prompt: str, continuation: str, image: Image) -> float:
        pass


from .blip2 import Blip2  # noqa: E402
from .gpt import GPT  # noqa: E402


def load_model(model_name: str) -> TextGenerationModel:
    if (
        model_name.startswith("gpt")
        or "davinci" in model_name
        or "curie" in model_name
        or "babbage" in model_name
        or "ada" in model_name
        or "cushman" in model_name
    ):
        return GPT(model_name)
    elif model_name in [
        "blip2-opt-2.7b",
        "blip2-opt-6.7b",
        "blip2-flan-t5-xl",
        "blip2-flan-t5-xxl",
    ]:
        return Blip2(model_name)
    else:
        raise ValueError


__all__ = ["TextGenerationModel", "Blip2", "GPT", "load_model"]
