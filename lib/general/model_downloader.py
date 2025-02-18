import os

from huggingface_hub import hf_hub_download

import folder_paths
from comfy.utils import load_torch_file


def add_model_folder_path(type_name: str):
    folder_paths.add_model_folder_path(type_name, os.path.join(folder_paths.models_dir, type_name))

def get_model_path_or_none(type_name: str, model_name: str) -> str | None:
    return folder_paths.get_full_path(type_name, model_name)

def get_target_model_path(type_name: str, model_name: str) -> str:
    """
    Should output something like comfyui/models/type_name/model_name
    :param type_name: Model type name
    :param model_name: Model file name
    :return: The full path
    """
    return os.path.join(folder_paths.models_dir, type_name, model_name)

def download_model_file(type_name: str, model_name: str, repo: str, file: str) -> str:
    """
    Download a single model file.
    :param type_name: The model type name to use, also a subdirectory
    :param model_name: The model name to save as
    :param repo: The hugginface repository to download the file from.
    :param file: The file to download from the repository.
    :return: The path to the downloaded model (prefers local if present)
    """
    hf_hub_download(repo, file, local_dir=get_target_model_path(type_name, model_name))
