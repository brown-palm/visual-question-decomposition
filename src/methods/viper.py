from dataclasses import dataclass
from PIL import Image

from src.methods import VQAModel, VQAModelResult
from src.models import TextGenerationModel, load_model


def generate_code(
    lm: TextGenerationModel, prompt_mode: str, question: str, choices: str = None
) -> str:
    if choices is None:
        fn_signature = "execute_command(image) -> str"
        query_prompt = f"# {question}"
    else:
        if prompt_mode == "only_queries":
            fn_signature = "execute_command(image) -> str"
        else:
            fn_signature = "execute_command(image, possible_answers) -> str"
        query_prompt = "\n".join([f"# {question}", f"# possible answers : {str(choices)}"])

    with open(f"src/methods/prompts/viper/{prompt_mode}.txt", "r") as f:
        base_prompt = f.read()

    prompt = base_prompt.replace("INSERT_QUERY_HERE", query_prompt).replace(
        "INSERT_SIGNATURE_HERE", fn_signature
    )

    response = lm.generate_text(
        input_text=prompt, image=None, max_new_tokens=512, stop="\n\n", temperature=0.0
    )

    if lm.model_name.startswith("gpt-3.5-turbo") or lm.model_name.startswith("gpt-4"):
        code = response
    else:
        code = f"def {fn_signature}:\n{response}"

    return code


@dataclass
class ViperModelResult(VQAModelResult):
    generated_code: str
    execution_results: dict


class ViperVQAModel(VQAModel):
    def __init__(self, lm_type: str, prompt_mode: str = "full_api"):
        self.lm = load_model(lm_type)
        from viper.execution import ViperExecutionModel

        self.viper_model = ViperExecutionModel()
        self.prompt_mode = prompt_mode

    def direct_answer(self, question: str, image: Image) -> VQAModelResult:
        return self.multiple_choice(question, None, image)

    def multiple_choice(self, question: str, choices: list[str], image: Image) -> VQAModelResult:
        code = generate_code(self.lm, self.prompt_mode, question, choices)
        execution_results = self.viper_model.execute_code(image, code, choices)
        prediction = str(execution_results.get("__return__", ""))
        return ViperModelResult(
            prediction=prediction,
            generated_code=code,
            execution_results=execution_results,
        )
