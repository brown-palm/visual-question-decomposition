"""
Adding a new functionality is easy. Just implement your new model as a subclass of BaseModel.
The code will make the rest: it will make it available for the processes to call by using
process(name, *args, **kwargs), where *args and **kwargs are the arguments of the models process() method.
"""

import abc
import contextlib
import openai
import os
import re
import timeit
import torch
import warnings
from PIL import Image
from collections import Counter
from itertools import chain
from torch import hub
from torch.nn import functional as F
from torchvision import transforms
from typing import Union

from .config import config
from . import utils


openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------------- Base abstract model --------------------------- #

class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}' if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]

# ------------------------------ Specific models ---------------------------- #


class DepthEstimationModel(BaseModel):
    name = 'depth'

    def __init__(self, gpu_number=0, model_type='DPT_Large'):
        super().__init__(gpu_number)
        warnings.simplefilter("ignore")
        # Model options: MiDaS_small, DPT_Hybrid, DPT_Large
        depth_estimation_model = hub.load('intel-isl/MiDaS', model_type, pretrained=True).to(self.dev)
        depth_estimation_model.eval()

        midas_transforms = hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.depth_estimation_model = depth_estimation_model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """Estimate depth map"""
        image_numpy = image.cpu().permute(1, 2, 0).numpy() * 255
        input_batch = self.transform(image_numpy).to(self.dev)
        prediction = self.depth_estimation_model(input_batch)
        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_numpy.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        # We compute the inverse because the model returns inverse depth
        to_return = 1 / prediction
        to_return = to_return.cpu()
        return to_return  # To save: plt.imsave(path_save, prediction.cpu().numpy())


class GLIPModel(BaseModel):
    name = 'glip'

    def __init__(self, gpu_number=0, *args):
        BaseModel.__init__(self, gpu_number)

        with contextlib.redirect_stderr(open(os.devnull, "w")):  # Do not print nltk_data messages when importing
            from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
                create_positive_map_label_to_token_from_positive_map

        working_dir = f'{hub.get_dir()}/viper/glip/'
        config_file = working_dir + "configs/glip_Swin_L.yaml"
        weight_file = working_dir + "checkpoints/glip_large_model.pth"

        class OurGLIPDemo(GLIPDemo):

            def __init__(self, dev, *args_demo):

                kwargs = {
                    'min_image_size': 800,
                    'confidence_threshold': config['glip_detect_threshold'],
                    'show_mask_heatmaps': False
                }

                self.dev = dev

                from maskrcnn_benchmark.config import cfg

                # manual override some options
                cfg.local_rank = 0
                cfg.num_gpus = 1
                cfg.merge_from_file(config_file)
                cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
                cfg.merge_from_list(["MODEL.DEVICE", self.dev])

                with torch.cuda.device(self.dev):
                    from transformers.utils import logging
                    logging.set_verbosity_error()
                    GLIPDemo.__init__(self, cfg, *args_demo, **kwargs)
                if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                    plus = 1
                else:
                    plus = 0
                self.plus = plus
                self.color = 255

            @torch.no_grad()
            def compute_prediction(self, original_image, original_caption, custom_entity=None):
                image = self.transforms(original_image)
                # image = [image, image.permute(0, 2, 1)]
                image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
                image_list = image_list.to(self.dev)
                # caption
                if isinstance(original_caption, list):

                    if len(original_caption) > 40:
                        all_predictions = None
                        for loop_num, i in enumerate(range(0, len(original_caption), 40)):
                            list_step = original_caption[i:i + 40]
                            prediction_step = self.compute_prediction(original_image, list_step, custom_entity=None)
                            if all_predictions is None:
                                all_predictions = prediction_step
                            else:
                                # Aggregate predictions
                                all_predictions.bbox = torch.cat((all_predictions.bbox, prediction_step.bbox), dim=0)
                                for k in all_predictions.extra_fields:
                                    all_predictions.extra_fields[k] = \
                                        torch.cat((all_predictions.extra_fields[k],
                                                   prediction_step.extra_fields[k] + loop_num), dim=0)
                        return all_predictions

                    # we directly provided a list of category names
                    caption_string = ""
                    tokens_positive = []
                    seperation_tokens = " . "
                    for word in original_caption:
                        tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                        caption_string += word
                        caption_string += seperation_tokens

                    tokenized = self.tokenizer([caption_string], return_tensors="pt")
                    # tokens_positive = [tokens_positive]  # This was wrong
                    tokens_positive = [[v] for v in tokens_positive]

                    original_caption = caption_string
                    # print(tokens_positive)
                else:
                    tokenized = self.tokenizer([original_caption], return_tensors="pt")
                    if custom_entity is None:
                        tokens_positive = self.run_ner(original_caption)
                    # print(tokens_positive)
                # process positive map
                positive_map = create_positive_map(tokenized, tokens_positive)

                positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map,
                                                                                                   plus=self.plus)
                self.positive_map_label_to_token = positive_map_label_to_token
                tic = timeit.time.perf_counter()

                # compute predictions
                predictions = self.model(image_list, captions=[original_caption],
                                            positive_map=positive_map_label_to_token)
                predictions = [o.to(self.cpu_device) for o in predictions]
                # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

                # always single image is passed at a time
                prediction = predictions[0]

                # reshape prediction (a BoxList) into the original image size
                height, width = original_image.shape[-2:]
                # if self.tensor_inputs:
                # else:
                #     height, width = original_image.shape[:-1]
                prediction = prediction.resize((width, height))

                if prediction.has_field("mask"):
                    # if we have masks, paste the masks in the right position
                    # in the image, as defined by the bounding boxes
                    masks = prediction.get_field("mask")
                    # always single image is passed at a time
                    masks = self.masker([masks], [prediction])[0]
                    prediction.add_field("mask", masks)

                return prediction

            @staticmethod
            def to_left_right_upper_lower(bboxes):
                return [(bbox[1], bbox[3], bbox[0], bbox[2]) for bbox in bboxes]

            @staticmethod
            def to_xmin_ymin_xmax_ymax(bboxes):
                # invert the previous method
                return [(bbox[2], bbox[0], bbox[3], bbox[1]) for bbox in bboxes]

            @staticmethod
            def prepare_image(image):
                image = image[[2, 1, 0]]  # convert to bgr for opencv-format for glip
                return image

            @torch.no_grad()
            def forward(self, image: torch.Tensor, obj: Union[str, list], return_labels: bool = False,
                        confidence_threshold=None):

                if confidence_threshold is not None:
                    original_confidence_threshold = self.confidence_threshold
                    self.confidence_threshold = confidence_threshold

                # if isinstance(object, list):
                #     object = ' . '.join(object) + ' .' # add separation tokens
                image = self.prepare_image(image)

                # Avoid the resizing creating a huge image in a pathological case
                ratio = image.shape[1] / image.shape[2]
                ratio = max(ratio, 1 / ratio)
                original_min_image_size = self.min_image_size
                if ratio > 10:
                    self.min_image_size = int(original_min_image_size * 10 / ratio)
                    self.transforms = self.build_transform()

                with torch.cuda.device(self.dev):
                    inference_output = self.inference(image, obj)

                bboxes = inference_output.bbox.cpu().numpy().astype(int)
                # bboxes = self.to_left_right_upper_lower(bboxes)

                if ratio > 10:
                    self.min_image_size = original_min_image_size
                    self.transforms = self.build_transform()

                bboxes = torch.tensor(bboxes)

                # Convert to [left, lower, right, upper] instead of [left, upper, right, lower]
                height = image.shape[-2]
                bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)

                if confidence_threshold is not None:
                    self.confidence_threshold = original_confidence_threshold
                if return_labels:
                    # subtract 1 because it's 1-indexed for some reason
                    return bboxes, inference_output.get_field("labels").cpu().numpy() - 1
                return bboxes

        self.glip_demo = OurGLIPDemo(*args, dev=self.dev)

    def forward(self, *args, **kwargs):
        return self.glip_demo.forward(*args, **kwargs)


class GPT3Model(BaseModel):
    name = 'gpt3'
    requires_gpu = False

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)
        self.qa_prompt = utils.gpt3_qa
        self.temperature = config['gpt3']['temperature']
        self.n_votes = config['gpt3']['n_votes']
        self.model = config['gpt3']['model']

    # initial cleaning for reference QA results
    @staticmethod
    def process_answer(answer):
        answer = answer.lstrip()  # remove leading spaces (our addition)
        answer = answer.replace('.', '').replace(',', '').lower()
        to_be_removed = {'a', 'an', 'the', 'to', ''}
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return ' '.join(answer_list)

    @staticmethod
    def get_union(lists):
        return list(set(chain.from_iterable(lists)))

    @staticmethod
    def most_frequent(answers):
        answer_counts = Counter(answers)
        return answer_counts.most_common(1)[0][0]

    def get_qa(self, prompts, prompt_base: str=None) -> list[str]:
        if prompt_base is None:
            prompt_base = self.qa_prompt
        prompts_total = []
        for p in prompts:
            question = p
            prompts_total.append(prompt_base.format(question))
        response = self.get_qa_fn(prompts_total)
        if self.n_votes > 1:
            response_ = []
            for i in range(len(prompts)):
                if self.model == 'chatgpt':
                    resp_i = [r['message']['content']
                              for r in response['choices'][i * self.n_votes:(i + 1) * self.n_votes]]
                else:
                    resp_i = [r['text'] for r in response['choices'][i * self.n_votes:(i + 1) * self.n_votes]]
                response_.append(self.most_frequent(resp_i))
            response = response_
        else:
            if self.model == 'chatgpt':
                response = [r['message']['content'] for r in response['choices']]
            else:
                response = [self.process_answer(r["text"]) for r in response['choices']]
        return response

    def get_qa_fn(self, prompt):
        response = self.query_gpt3(prompt, model=self.model, max_tokens=5, logprobs=1, stream=False,
                                   stop=["\n", "<|endoftext|>"])
        return response

    def get_general(self, prompts) -> list[str]:
        if self.model == "chatgpt":
            raise NotImplementedError
        response = self.query_gpt3(prompts, model=self.model, max_tokens=256, top_p=1, frequency_penalty=0,
                                   presence_penalty=0)
        response = [r["text"] for r in response['choices']]
        return response

    def query_gpt3(self, prompt, model="text-davinci-003", max_tokens=16, logprobs=None, stream=False,
                   stop=None, top_p=1, frequency_penalty=0, presence_penalty=0):
        if model == "chatgpt":
            messages = [{"role": "user", "content": p} for p in prompt]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                logprobs=logprobs,
                temperature=self.temperature,
                stream=stream,
                stop=stop,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=self.n_votes,
            )
        return response

    def forward(self, prompt, process_name):
        prompt = [prompt]
        
        if process_name == 'gpt3_qa':
            # if items in prompt are tuples, then we assume it is a question and context
            if isinstance(prompt[0], tuple) or isinstance(prompt[0], list):
                prompt = [question.format(context) for question, context in prompt]

        to_compute = None
        results = []

        if len(prompt) > 0:
            if process_name == 'gpt3_qa':
                results = self.get_qa(prompt)
            else:  # 'gpt3_general', general prompt, has to be given all of it
                results = self.get_general(prompt)

        results = results[0]
        return results

    @classmethod
    def list_processes(cls):
        return ['gpt3_' + n for n in ['qa', 'general']]


class BLIPModel(BaseModel):
    name = 'blip'
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2  # The queue has additionally the time it is executing the previous forward pass

    def __init__(self, gpu_number=0, half_precision=config['blip_half_precision'],
                 blip_v2_model_type=config['blip_v2_model_type']):
        super().__init__(gpu_number)

        # from lavis.models import load_model_and_preprocess
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        # https://huggingface.co/models?sort=downloads&search=Salesforce%2Fblip2-
        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']

        with warnings.catch_warnings(), torch.cuda.device(self.dev):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}

            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
            # Device_map must be sequential for manual GPU selection
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    f"Salesforce/{blip_v2_model_type}", load_in_8bit=half_precision,
                    torch_dtype=torch.float16 if half_precision else "auto",
                    device_map="sequential", max_memory=max_memory
                )
            except Exception as e:
                # Clarify error message. The problem is that it tries to load part of the model to disk.
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e

        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.half_precision = half_precision
        self.max_words = 50

    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.dev, torch.float16)
        generated_ids = self.model.generate(**inputs, length_penalty=1., num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = [cap.strip() for cap in
                          self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        return generated_text
    
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self, image, question=None, task='caption'):
        if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
            image = [im * 255 for im in image]

        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []

        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))

        return response


class XVLMModel(BaseModel):
    name = 'xvlm'

    def __init__(self, gpu_number=0):

        from .xvlm.xvlm import XVLMBase
        from transformers import BertTokenizer

        super().__init__(gpu_number)

        image_res = 384
        self.max_words = 30
        config_xvlm = {
            'image_res': image_res,
            'patch_size': 32,
            'text_encoder': 'bert-base-uncased',
            'block_num': 9,
            'max_tokens': 40,
            'embed_dim': 256,
        }

        vision_config = {
            'vision_width': 1024,
            'image_res': 384,
            'window_size': 12,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32]
        }
        with warnings.catch_warnings():
            model = XVLMBase(config_xvlm, use_contrastive_loss=True, vision_config=vision_config)
            checkpoint = torch.load(f'{hub.get_dir()}/viper/xvlm/retrieval_mscoco_checkpoint_9.pth', map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) > 0:
            print('XVLM Missing keys: ', msg.missing_keys)

        model = model.to(self.dev)
        model.eval()

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

    @torch.no_grad()
    def score(self, images, texts):

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(images, list):
            images = [images]

        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.dev)

        texts = [self.pre_caption(text, self.max_words) for text in texts]
        text_input = self.tokenizer(texts, padding='longest', return_tensors="pt").to(self.dev)

        image_embeds, image_atts = self.model.get_vision_embeds(images)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.model.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.model.get_features(image_embeds, text_embeds)
        logits = image_feat @ text_feat.t()

        return logits

    @torch.no_grad()
    def binary_score(self, image, text, negative_categories):
        # Compare with a pre-defined set of negatives
        texts = [text] + negative_categories
        sim = 100 * self.score(image, texts)[0]
        res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                   sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    def forward(self, image, text, task='score', negative_categories=None):
        if task == 'score':
            score = self.score(image, text)
        else:  # binary
            score = self.binary_score(image, text, negative_categories=negative_categories)
        return score.cpu()
