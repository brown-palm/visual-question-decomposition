from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from . import TextGenerationModel


@dataclass
class Blip2(TextGenerationModel):
    def __post_init__(self):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.processor = Blip2Processor.from_pretrained(f"Salesforce/{self.model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/{self.model_name}",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
        )

    def generate_text(
        self,
        input_text: str = None,
        image: Image = None,
        max_new_tokens: int = 10,
        stop: str = "\n",
        temperature: float = 1.0,
    ) -> str:
        inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        output = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            length_penalty=-1,
            top_p=0.9,
            repetition_penalty=1.0,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]
        output_text = self.processor.decode(output, skip_special_tokens=True).strip()

        if stop in output_text:
            output_text = output_text[: output_text.find(stop)]

        return output_text

    def get_continuation_likelihood(self, prompt: str, continuation: str, image: Image) -> float:
        assert self.model_name.startswith("blip2-flan-t5")

        inputs = self.processor(images=image, text=(prompt + continuation), return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        inputs["labels"] = inputs["input_ids"]
        input_ids = inputs["input_ids"][0]
        tokens = [self.processor.decode([t]) for t in input_ids]

        logits = self.model(**inputs).logits[0]
        logprobs = F.log_softmax(logits, dim=1)
        logprobs = [logprobs[i, inputs["input_ids"][0][i]] for i in range(len(tokens))]

        return self._continuation_wbl_likelihood(prompt, continuation, tokens, logprobs)
