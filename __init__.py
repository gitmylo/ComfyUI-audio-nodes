import os.path

import folder_paths
from .lib.general.model_downloader import add_model_folder_path

add_model_folder_path("bark") # Bark models are downloaded here
add_model_folder_path("bark_speakers", os.path.join(folder_paths.models_dir, "bark", "speakers")) # Bark speaker npz files
add_model_folder_path("bark_quantizers", os.path.join(folder_paths.models_dir, "bark", "quantizers")) # Bark Hubert Quantizers

# Node imports go here
from .nodes import nodes

NODE_CLASS_MAPPINGS = {node.name: node for node in nodes}
NODE_DISPLAY_NAME_MAPPINGS = {node.name: node.display_name for node in nodes}