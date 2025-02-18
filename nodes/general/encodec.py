from dataclasses import dataclass
from typing import Any
from encodec.model import EncodecModel

from ...util.categories import category
from ...nodes.basenode import BaseNode

@dataclass
class EncodecModelRules:
    cpu: bool
    offload: bool

    model: EncodecModel = None

    def load(self):
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        model.to("cpu" if (self.offload or self.cpu) else "cuda")
        self.model = model

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
                "cpu": ("BOOLEAN",),
                "offload": ("BOOLEAN",)
            }
        }

    def load(self, cpu: bool, offload: bool):
        model = EncodecModelRules(cpu, offload)
        model.load()

        # TODO: Ensure the model can be unloaded automatically

        return (None,)


nodes = [EncodecLoader]