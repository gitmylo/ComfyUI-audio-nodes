from dataclasses import dataclass
from typing import Any

import torch
from encodec.model import EncodecModel

from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from ...util.categories import category
from ...nodes.basenode import BaseNode

import comfy.model_management as mm

@dataclass
class EncodecModelRules:
    cpu: bool

    model: EncodecModel = None
    patcher: ModelPatcher = None

    def load(self):
        main_device = mm.get_torch_device() if not self.cpu else torch.device("cpu")
        offload_device = mm.unet_offload_device()

        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        model.to(main_device)

        patcher = ModelPatcher(model, main_device, offload_device)  # Allows comfy to unload and offload
        mm.load_model_gpu(patcher)
        self.patcher = patcher

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
                "cpu": ("BOOLEAN",)
            }
        }

    def load(self, cpu: bool):
        model = EncodecModelRules(cpu)
        model.load()

        return (model,)


nodes = [EncodecLoader]