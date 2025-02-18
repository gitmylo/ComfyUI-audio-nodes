from comfy_execution.graph_utils import GraphBuilder
from ...nodes.basenode import BaseNode
from ...util.categories import category

class BarkLoader(BaseNode):
    name = "bark_loader"
    display_name = "Load bark model(s)"

    RETURN_NAMES = ("text", "coarse", "fine")
    RETURN_TYPES = ("BarkTextModel", "BarkCoarseModel", "BarkFineModel")
    CATEGORY = category("bark")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (["large", "small", "none"],),
                "coarse": (["large", "small", "none"],),
                "fine": (["large", "small", "none"],),
                "half": ("BOOLEAN",),
                "offload_text": ("BOOLEAN",),
                "offload_coarse": ("BOOLEAN",),
                "offload_fine": ("BOOLEAN",)
            }
        }

    def load(self, text, coarse, fine, half, offload_text, offload_coarse, offload_fine):
        # Using https://huggingface.co/GitMylo/bark-safetensors since it's recommended to use safetensors

        return None, None, None

nodes = [BarkLoader]