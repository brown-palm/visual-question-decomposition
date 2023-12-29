import time
from dataclasses import dataclass

from PIL import Image

from . import TextGenerationModel


@dataclass
class GPT(TextGenerationModel):
    organization: str = None

    def __post_init__(self):
        import openai

        self.openai = openai

    def generate_text(
        self,
        input_text: str,
        image: Image.Image = None,
        max_new_tokens: int = 256,
        stop: str = "\n",
        temperature: float = 0.0,
    ) -> str:
        while True:
            try:
                if (
                    self.model_name.startswith("gpt-")
                    and self.model_name != "gpt-3.5-turbo-instruct"
                ):
                    response = self.openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": input_text}],
                        max_tokens=max_new_tokens,
                        stop=stop,
                        temperature=temperature,
                    )["choices"][0]["message"]["content"]
                else:
                    response = self.openai.Completion.create(
                        model=self.model_name,
                        prompt=input_text,
                        max_tokens=max_new_tokens,
                        stop=stop,
                        temperature=temperature,
                    )["choices"][0]["text"]
                break

            except (
                self.openai.error.ServiceUnavailableError,
                self.openai.error.APIError,
                self.openai.error.RateLimitError,
                self.openai.error.APIConnectionError,
            ) as e:
                print("API error:", e)
                time.sleep(10)

        return response

    def get_continuation_likelihood(
        self, prompt: str, continuation: str, image: Image = None
    ) -> float:

        assert (
            self.model_name.startswith("gpt-") is False
            or self.model_name == "gpt-3.5-turbo-instruct"
        ), "Chat Completions models do not support logprobs"

        while True:
            try:
                logprobs = self.openai.Completion.create(
                    model=self.model_name,
                    prompt=(prompt + continuation),
                    max_tokens=0,
                    echo=True,
                    logprobs=1,
                )["choices"][0]["logprobs"]
                break
            except (
                self.openai.error.ServiceUnavailableError,
                self.openai.error.APIError,
                self.openai.error.RateLimitError,
                self.openai.error.APIConnectionError,
            ) as e:
                print("API error:", e)
                time.sleep(10)

        tokens = logprobs["tokens"]
        logprobs = logprobs["token_logprobs"]

        return self._continuation_wbl_likelihood(prompt, continuation, tokens, logprobs)
