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
- `VQA_DIR`, `GQA_DIR`, `OKVQA_DIR`, `AOKVQA_DIR`, `COCO_DIR`: for [storing datasets](#download-datasets)

### Download Viper models

Download models to `$TORCH_HOME/hub/viper` (usually `~/.cache/torch/hub/viper`).

```bash
python -m viper.download_models
```

### Download datasets

Datasets are saved to `$VQA_DIR`, `$GQA_DIR`, `$OKVQA_DIR`, `$AOKVQA_DIR`, and `$COCO_DIR` (by default: `./datasets/*`).

```bash
# download all datasets
python -m src.download_data

# download a specific dataset
python -m src.download_data dataset:{vqav2,gqa,okvqa,aokvqa,coco}
```

# Running experiments

Run the core experiments (with default settings from our paper):

```bash
python experiments/vqa.py \
dataset:{vqav2,gqa,okvqa,aokvqa,scienceqa} \
method:{e2e,viper,successive}
```

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

Unfortunately, the default GPT models (`code-davinci-002`, `text-davinci-002`, `text-davinci-003`) used in ViperGPT and our paper are (or will shortly be) [deprecated](https://platform.openai.com/docs/deprecations). For reproducibility and best practices, we strongly recommend using open-source LLMs in your future research.

We instead provide the options (in the above CLI settings) to use different GPT models. However, our work is designed around the legacy [Completions API](https://platform.openai.com/docs/api-reference/completions), so your milage may vary. If you use a Chat model (e.g. `gpt-4`) with ViperGPT, you may have to adjust the prompt ([example](https://github.com/cvlab-columbia/viper/blob/main/prompts/chatapi.prompt)). The Successive method, GPT-based evaluation metric, and MC evaluation of ViperGPT strictly require the Completions API and will not work.

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
