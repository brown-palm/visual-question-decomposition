from typing import List, Dict
import ast
import pickle
from dataclasses import dataclass

from torchvision import transforms
from PIL import Image
import functools
import inspect
import traceback

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


@dataclass
class BoundingBox:
    left: int
    lower: int
    right: int
    upper: int


def _patches_to_bboxes(o):
    if isinstance(o, image_patch.ImagePatch):
        return BoundingBox(o.left, o.lower, o.right, o.upper)
    elif isinstance(o, tuple):
        return tuple(_patches_to_bboxes(x) for x in o)
    elif isinstance(o, list):
        return [_patches_to_bboxes(x) for x in o]
    elif isinstance(o, dict):
        return {k:_patches_to_bboxes(v) for k, v in o.items()}
    else:
        return o


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
        self.llm_query_fn = functools.partial(image_patch.llm_query, self.forward)
        self.best_image_match_fn = functools.partial(image_patch.best_image_match, config)


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

    def execute_code(self, image: Image.Image, code: str, possible_answers: List[str] = None) -> Dict:
        import numpy as np
        import math
        ImagePatch = self.image_patch_class
        llm_query = self.llm_query_fn
        best_image_match = self.best_image_match_fn
        from .image_patch import (
            distance,
            bool_to_yesno,
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

        # Cast image patches to BoundingBox class
        execution_result = {k:_patches_to_bboxes(v) for k, v in execution_result.items()}

        # Remove un-picklable elements from execution_result
        def is_picklable(obj):
            try:
                pickle.dumps(obj)
            except pickle.PicklingError:
                return False
            return True

        execution_result = {k:v for k, v in execution_result.items() if is_picklable(v)}

        return execution_result
