import io
import os
import tarfile
import zipfile
from typing import Literal, Union

import datasets
import requests
import tyro


def download_zip(url: str, output_dir: Union[str, os.PathLike]):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output_dir)


def download_tar_gz(url: str, output_dir: Union[str, os.PathLike]):
    r = requests.get(url)
    tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz").extractall(output_dir)


def main(dataset: Literal["all", "vqav2", "gqa", "okvqa", "aokvqa", "coco", "scienceqa"] = "all"):
    download_all = dataset == "all"

    if download_all or dataset == "vqav2":
        vqa_dir = os.environ["VQA_DIR"]
        print(f'Downloading VQAv2 to {vqa_dir}')

        for filename in [
            "v2_Annotations_Train_mscoco",
            "v2_Annotations_Val_mscoco",
            "v2_Questions_Train_mscoco",
            "v2_Questions_Val_mscoco",
            "v2_Questions_Test_mscoco",
        ]:
            download_zip(f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/{filename}.zip", vqa_dir)

    if download_all or dataset == "gqa":
        gqa_dir = os.environ["GQA_DIR"]
        print(f'Downloading GQA to {gqa_dir}')

        for filename in ["sceneGraphs", "questions1.2", "images"]:
            download_zip(
                f"https://downloads.cs.stanford.edu/nlp/data/gqa/{filename}.zip",
                os.path.join(gqa_dir, filename),
            )

    if download_all or dataset == "okvqa":
        okvqa_dir = os.environ["OKVQA_DIR"]
        print(f'Downloading OK-VQA to {okvqa_dir}')

        for filename in [
            "mscoco_train2014_annotations.json",
            "mscoco_val2014_annotations.json",
            "OpenEnded_mscoco_train2014_questions.json",
            "OpenEnded_mscoco_val2014_questions.json",
        ]:
            download_zip(f"https://okvqa.allenai.org/static/data/{filename}.zip", okvqa_dir)

    if download_all or dataset == "aokvqa":
        aokvqa_dir = os.environ["AOKVQA_DIR"]
        print(f'Downloading A-OKVQA to {aokvqa_dir}')

        download_tar_gz(
            "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz",
            aokvqa_dir,
        )

    if download_all or dataset == "coco":
        coco_dir = os.environ["COCO_DIR"]
        print(f'Downloading A-OKVQA to {coco_dir}')

        for filename in [
            "train2014",
            "val2014",
            "test2014",
            "test2015",
            "train2017",
            "val2017",
            "test2017",
        ]:
            download_zip(f"http://images.cocodataset.org/zips/{filename}.zip", coco_dir)

    if download_all or dataset == "scienceqa":
        print('Downloading ScienceQA')
        datasets.load_dataset("derek-thomas/ScienceQA")

if __name__ == "__main__":
    tyro.cli(main)
