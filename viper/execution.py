from typing import Dict
import ast

from torchvision import transforms
from PIL import Image
import functools
import inspect

from .config import config
from . import image_patch, vision_models


class CompletedExecution(Exception):
    pass


# Replaces Return nodes with Assign in ast
class ReturnTransformer(ast.NodeTransformer):
    def visit_Return(self, node):
        return [
            ast.Assign(
                targets=[ast.Name(id="__return__", ctx=ast.Store())], value=node.value
            ),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="CompletedExecution", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            ),
        ]


class ViperExecutionModel:
    # specify model -> GPU map
    def __init__(self):
        self.model_instances = dict()
        self.to_batch = dict()

        for _, model_class in inspect.getmembers(vision_models, inspect.isclass):
            if vision_models.BaseModel not in model_class.__bases__:
                continue

            for p in model_class.list_processes():
                if config['load_models'].get(p, False):
                    self.model_instances[p] = model_class(gpu_number=0)
                    self.to_batch[p] = model_class.to_batch

        self.image_patch_class = functools.partial(image_patch.ImagePatch, config, self.forward)


    def forward(self, model_name, *args, **kwargs):
        model_instance = self.model_instances[model_name]

        if len(model_instance.list_processes()) > 1:
            kwargs["process_name"] = model_name

        if self.to_batch[model_name]:
            # Batchify the input. Model expects a batch. And later un-batchify the output.
            args = [[arg] for arg in args]
            kwargs = {k: [v] for k, v in kwargs.items()}

            # The defaults that are not in args or kwargs, also need to listify
            full_arg_spec = inspect.getfullargspec(model_instance.forward)
            if full_arg_spec.defaults is None:
                default_dict = {}
            else:
                default_dict = dict(
                    zip(
                        full_arg_spec.args[-len(full_arg_spec.defaults) :],
                        full_arg_spec.defaults,
                    )
                )
            non_given_args = full_arg_spec.args[1:][len(args) :]
            non_given_args = set(non_given_args) - set(kwargs.keys())
            for arg_name in non_given_args:
                kwargs[arg_name] = [default_dict[arg_name]]

        try:
            out = model_instance.forward(*args, **kwargs)
            if self.to_batch[model_name]:
                out = out[0]
        except Exception as e:
            print(f"Error in {model_name} model:", e)
            out = None

        return out

    def execute_code(self, image: Image.Image, code: str) -> Dict:
        ImagePatch = self.image_patch_class
        import math
        from .image_patch import (
            best_image_match,
            distance,
            bool_to_yesno,
            llm_query,
            coerce_to_numeric,
        )

        execution_result = {}

        try:
            # Parse the code into an AST (abstract syntax tree)
            tree = ast.parse(code)
            tree = ReturnTransformer().visit(tree)
            tree = ast.fix_missing_locations(tree)

            # Execute each top-level node, recording function locals

            for node in tree.body[0].body:
                node = ast.Module(body=[node], type_ignores=[])
                compiled_node = compile(node, filename="<ast>", mode="exec")
                exec(
                    compiled_node,
                    {**globals(), **locals(), **execution_result},
                    execution_result,
                )

        except CompletedExecution:
            pass
        except:
            execution_result["__error__"] = traceback.format_exc()
            print("Execution Error: " + execution_result["__error__"])

        return execution_result
