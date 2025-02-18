from .lib.general.model_downloader import add_model_folder_path

add_model_folder_path("bark") # Bark models are downloaded here

# Node imports go here
from .nodes import nodes

NODE_CLASS_MAPPINGS = {node.name: node for node in nodes}
NODE_DISPLAY_NAME_MAPPINGS = {node.name: node.display_name for node in nodes}