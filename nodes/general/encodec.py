from ...util.categories import category
from ...nodes.basenode import BaseNode

class EncodecLoader(BaseNode):
    name = "encodec_loader"
    display_name = "Load encodec model"

    RETURN_NAMES = ("model",)
    RETURN_TYPES = ("EncodecModel",)
    CATEGORY = category("general")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offload": ("BOOLEAN",)
            }
        }

    def load(self, offload):
        # Using https://huggingface.co/GitMylo/bark-safetensors since it's recommended to use safetensors anyway
        return (None,)


nodes = [EncodecLoader]