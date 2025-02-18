import dataclasses
import json
from time import sleep

from typing import Literal
from dataclasses import dataclass

import safetensors.torch
import torch
from transformers import BertTokenizer

import comfy.utils
from lib.bark.model import GPTConfig, GPT
from lib.bark.model_fine import FineGPTConfig, FineGPT
from lib.general.model_downloader import download_model_file
from ...nodes.basenode import BaseNode
from ...util.categories import category

BarkModelVariant = Literal["large", "small", "none"]

@dataclass
class BarkModelRules:
    name: str
    variant: BarkModelVariant
    offload: bool
    use_cpu: bool
    half: bool

    model: GPT | FineGPT = None
    tokenizer: BertTokenizer = None # Only used for text

    def load(self):
        if self.variant == "none":
            return

        cc = None
        mc = None
        if self.name == "text":
            cc = GPTConfig
            mc = GPT
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        elif self.name == "coarse":
            cc = GPTConfig
            mc = GPT
        elif self.name == "fine":
            cc = FineGPTConfig
            mc = FineGPT
        else:
            raise NotImplementedError()

        base_name = self.name if self.variant == "large" else (self.name + "_small")
        checkpoint_name = self.name + ".safetensors"
        config_name = self.name + ".json"

        repo = "GitMylo/bark-safetensors"
        # (down)load config and checkpoint (Since the safetensors models are separate)
        config_path = download_model_file("bark", repo, config_name) # (down)load config
        model_path = download_model_file("bark", repo, checkpoint_name) # (down)load checkpoint

        with open(config_path, "r") as cfg:
            model_instance = mc(cc(**json.load(cfg)))

        selected_device = torch.device("cpu" if (self.offload or self.use_cpu) else "cuda")
        safetensors.torch.load_model(model_instance, model_path) # Loads on cpu initially
        model_instance.eval()
        model_instance.to(selected_device)
        self.model = model_instance


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
                "use_cpu": ("BOOLEAN",),
                "half": ("BOOLEAN",),
                "offload_text": ("BOOLEAN",),
                "offload_coarse": ("BOOLEAN",),
                "offload_fine": ("BOOLEAN",)
            }
        }

    def load(self, text: BarkModelVariant, coarse: BarkModelVariant, fine: BarkModelVariant, use_cpu: bool, half: bool, offload_text: bool, offload_coarse: bool, offload_fine: bool):
        # Using https://huggingface.co/GitMylo/bark-safetensors since it's recommended to use safetensors

        if use_cpu:
            # When using cpu only, offloading is useless
            offload_fine = False
            offload_coarse = False
            offload_text = False

        load_models = [BarkModelRules("text", text, offload_text, use_cpu, half), BarkModelRules("coarse", coarse, offload_coarse, use_cpu, half), BarkModelRules("fine", fine, offload_fine, use_cpu, half)]

        bar = comfy.utils.ProgressBar(len(load_models))
        for model in load_models:
            if model.variant != "none":
                print(f"(down)loading bark {model.name}")
            model.load()
            bar.update(1)

        # TODO: Ensure the models are properly registered so they can be cleared automatically when not needed.

        return tuple(load_models)

nodes = [BarkLoader]