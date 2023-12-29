# Analyzing Modular Approaches for Visual Question Decomposition

[Apoorv Khandelwal](https://apoorvkh.com), [Ellie Pavlick](https://cs.brown.edu/people/epavlick), and [Chen Sun](https://chensun.me)

EMNLP 2023

---

### Abstract

Modular neural networks without additional training have recently been shown to surpass end-to-end neural networks on challenging vision-language tasks. The latest such methods simultaneously introduce LLM-based code generation to build programs and a number of skill-specific, task-oriented modules to execute them. In this paper, we focus on ViperGPT and ask where its additional performance comes from and how much is due to the (state-of-art, end-to-end) BLIP-2 model it subsumes vs. additional symbolic components. To do so, we conduct a controlled study (comparing end-to-end, modular, and prompting-based methods across several VQA benchmarks). We find that ViperGPT's reported gains over BLIP-2 can be attributed to its selection of task-specific modules, and when we run ViperGPT using a more task-agnostic selection of modules, these gains go away. Additionally, ViperGPT retains much of its performance if we make prominent alterations to its selection of modules: e.g. removing or retaining only BLIP-2. Finally, we compare ViperGPT against a prompting-based decomposition strategy and find that, on some benchmarks, modular approaches significantly benefit by representing subtasks with natural language, instead of code.

![image](https://github.com/brown-palm/visual-question-decomposition/assets/7005565/0b9f2a42-e2c5-4f6c-8036-6fe5bff24068)


# Installation

### Hardware requirements

- 64G system memory
- High-memory Ampere GPU (e.g. 48GB Nvidia RTX A6000)

### Setup

You **must** run the following commands on your GPU machine, as certain dependencies require CUDA compilation.
We highly recommend using the *much faster* [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as a nearly-drop-in replacement for `conda`.

```bash
conda env create -f conda-lock.yml --prefix ./.venv
conda activate ./.venv
pdm install
```

### Environment Variables

You can adjust the environment variables in `.env`. If you make changes, run `conda activate ./.venv` again to reload these variables.

- `OPENAI_ORGANIZATION`: your OpenAI [organization ID](https://platform.openai.com/account/organization)
- `OPENAI_API_KEY`: your OpenAI [API key](https://platform.openai.com/api-keys)
- `TORCH_HOME`: for [downloading ViperGPT models](#download-viper-models)
- `VQA_DIR`, `GQA_DIR`, `OKVQA_DIR`, `AOKVQA_DIR`, `COCO_DIR`, `HF_HOME`: for [storing datasets](#download-datasets)

### Download Viper models

```bash
# download models to `$TORCH_HOME/hub/viper` (usually `~/.cache/torch/hub/viper`)
python -m viper.download_models
```

### Download datasets

```bash
# download all datasets
python -m src.data.download

# download a specific dataset
python -m src.data.download --dataset {vqav2,gqa,okvqa,aokvqa,coco,scienceqa}

## coco is required for vqav2, okvqa, and aokvqa
## scienceqa is saved to $HF_HOME/datasets/derek-thomas___science_qa
```

# Running experiments

Never forget to `conda activate ./.venv`.

Run the core experiments (with default settings from our paper):

```bash
python experiments/vqa.py \
dataset:{vqav2,gqa,okvqa,aokvqa,scienceqa} \
method:{e2e,viper,successive}
```

This repo uses [AI2 Tango](github.com/allenai/tango) for experiment tracking and caching.

### Additional Settings

Explore the `--help` menus for additional settings!

```bash
# For LLM evaluation options
python experiments/vqa.py --help

# For dataset arguments
python experiments/vqa.py dataset:<...> --help

# For method arguments
python experiments/vqa.py dataset:<...> method:<...> --help
```

Example:

```bash
python experiments/vqa.py --gpt-eval-model text-davinci-003 dataset:vqav2 --dataset.split val2014 --dataset.n 5 method:e2e --method.model-type blip2-flan-t5-xxl

# Output

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ instructgpt_acc ┃ vqav2_acc ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 80.0            │ 80.0      │
└─────────────────┴───────────┘
```

### Deprecated OpenAI Models

Unfortunately, the default GPT models (`code-davinci-002`, `text-davinci-002`, `text-davinci-003`) used in ViperGPT and our paper are (or will shortly be) [deprecated](https://platform.openai.com/docs/deprecations). Moreover, the legacy [Completions API](https://platform.openai.com/docs/api-reference/completions) is critical to several functions of this repository. You may work around these restrictions by specifying different GPT models and adjusting the prompts appropriately (e.g. see this [chat prompt for ViperGPT](https://github.com/cvlab-columbia/viper/blob/main/prompts/chatapi.prompt)), but your milage may vary. For reproducibility and best practices, we strongly recommend using open-source LLMs in your future research.

# Citation

```bibtex
@inproceedings{khandelwal2023:vqd,
    title        = {Analyzing Modular Approaches for Visual Question Decomposition},
    author       = {Apoorv Khandelwal and Ellie Pavlick and Chen Sun},
    year         = {2023},
    month        = {December},
    booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    pages        = {2590--2603}
}
```
