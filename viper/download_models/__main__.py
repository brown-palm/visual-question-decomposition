import os
import torch
from huggingface_hub import hf_hub_download


TORCH_HUB = torch.hub.get_dir()

MODEL_FILES = [
    "viper/glip/checkpoints/glip_large_model.pth",
    "viper/glip/configs/glip_Swin_L.yaml",
    "viper/xvlm/retrieval_mscoco_checkpoint_9.pth"
]

for f in MODEL_FILES:
    subfolder, filename = os.path.split(f)
    hf_hub_download(
        repo_id="apoorvkh/visual-question-decomposition",
        filename=filename,
        subfolder=subfolder,
        revision="main",
        local_dir=TORCH_HUB,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
