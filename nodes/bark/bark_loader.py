import dataclasses
import json
import os
from time import sleep

from typing import Literal
from dataclasses import dataclass

import numpy as np
import safetensors.torch
import torch
from transformers import BertTokenizer, HubertModel

import comfy.utils
import folder_paths
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from ...lib.bark.hubert.hubert_manager import HuBERTManager
from ...lib.bark.hubert.customtokenizer import CustomTokenizer
from ...lib.bark.hubert.pre_kmeans_hubert import CustomHubert
from ...lib.bark.model import GPTConfig, GPT
from ...lib.bark.model_fine import FineGPTConfig, FineGPT
from ...lib.general.model_downloader import download_model_file
from ...nodes.basenode import BaseNode
from ...util.categories import category

import comfy.model_management as mm

BarkModelVariant = Literal["large", "small", "none"]

@dataclass
class BarkModelRules:
    name: str
    variant: BarkModelVariant
    use_cpu: bool
    half: bool

    model: GPT | FineGPT = None
    patcher: ModelPatcher = None
    tokenizer: BertTokenizer = None # Only used for text

    def load(self):
        if self.variant == "none":
            return

        main_device = mm.get_torch_device() if not self.use_cpu else torch.device("cpu")
        offload_device = mm.unet_offload_device()

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

        if self.half:
            model_instance = model_instance.half()

        safetensors.torch.load_model(model_instance, model_path) # Loads on cpu initially
        model_instance.eval()
        model_instance.to(main_device)

        patcher = ModelPatcher(model_instance, main_device, offload_device) # Allows comfy to unload and offload
        mm.load_model_gpu(patcher)
        self.patcher = patcher

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
                "half": ("BOOLEAN",)
            }
        }

    def load(self, text: BarkModelVariant, coarse: BarkModelVariant, fine: BarkModelVariant, use_cpu: bool, half: bool):
        # Using https://huggingface.co/GitMylo/bark-safetensors since it's recommended to use safetensors

        load_models = [BarkModelRules("text", text, use_cpu, half), BarkModelRules("coarse", coarse, use_cpu, half), BarkModelRules("fine", fine, use_cpu, half)]

        bar = comfy.utils.ProgressBar(len(load_models))
        for model in load_models:
            model.load()
            bar.update(1)

        return tuple(load_models)

class LoadSpeakerNPZ(BaseNode):
    name = "load_speaker_npz"
    display_name = "Load bark speaker .npz"

    RETURN_NAMES = ("text", "coarse", "fine")
    RETURN_TYPES = ("BarkSemanticTokens", "EncodecCodeBooks", "EncodecCodeBooks")
    CATEGORY = category("bark")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (folder_paths.get_filename_list("bark_speakers"),)
            }
        }

    def load(self, file: str):
        speaker_path = folder_paths.get_full_path_or_raise("bark_speakers", file)
        speaker = np.load(speaker_path)
        return speaker["semantic_prompt"], speaker["coarse_prompt"], speaker["fine_prompt"]

class SaveSpeakerNPZ(BaseNode):
    name = "save_speaker_npz"
    display_name = "Save bark speaker .npz"

    RETURN_NAMES = ()
    RETURN_TYPES = ()
    CATEGORY = category("bark")
    FUNCTION = "save"

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "speaker"}),
                "semantic_prompt": ("BarkSemanticTokens",),
                "coarse_prompt": ("EncodecCodeBooks",),
                "fine_prompt": ("EncodecCodeBooks",)
            }
        }

    def save(self, name: str, semantic_prompt, coarse_prompt, fine_prompt):
        speaker_path = os.path.join(folder_paths.models_dir, "bark", "speakers", name)
        np.savez(speaker_path, semantic_prompt=semantic_prompt, coarse_prompt=coarse_prompt, fine_prompt=fine_prompt)
        return ()

@dataclass
class BarkHubertModelRules:
    cpu: bool

    model: CustomHubert = None
    patcher: ModelPatcher = None

    def load(self):
        main_device = mm.get_torch_device() if not self.cpu else torch.device("cpu")
        offload_device = mm.unet_offload_device()

        self.model = CustomHubert()
        self.patcher = ModelPatcher(self.model, main_device, offload_device)

class LoadHubertBark(BaseNode):
    name = "load_bark_hubert"
    display_name = "Load HuBERT model (bark)"

    RETURN_NAMES = ("model",)
    RETURN_TYPES = ("BarkHuBERTModel",)
    CATEGORY = category("bark/cloning")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "cpu": ("BOOLEAN",)
            }
        }

    def load(self, cpu: bool = True):
        model = BarkHubertModelRules(cpu)
        model.load()
        return (model,)

@dataclass
class BarkHubertQuantizerModelRules:
    cpu: bool
    model: str

    model: CustomTokenizer = None
    patcher: ModelPatcher = None

    def load(self):
        main_device = mm.get_torch_device() if not self.cpu else torch.device("cpu")
        offload_device = mm.unet_offload_device()

        model_file = HuBERTManager.make_sure_tokenizer_installed(self.model)
        self.model = CustomTokenizer.load_from_checkpoint(model_file)
        self.patcher = ModelPatcher(self.model, main_device, offload_device)

class LoadHubertQuantizerBark(BaseNode):
    name = "load_bark_hubert_quantizer"
    display_name = "Load HuBERT quantizer model (bark)"

    RETURN_NAMES = ("model",)
    RETURN_TYPES = ("BarkHuBERTQuantizerModel",)
    CATEGORY = category("bark/cloning")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "cpu": ("BOOLEAN",)
                "model": (["quantifier_hubert_base_ls960.pth", "quantifier_hubert_base_ls960_14.pth", "quantifier_V1_hubert_base_ls960_23.pth"],) # TODO: Load from online/local registry
            }
        }

    def load(self, model, cpu: bool = True):
        model = BarkHubertQuantizerModelRules(cpu, model)
        model.load()
        return (model,)

nodes = [BarkLoader, LoadSpeakerNPZ, SaveSpeakerNPZ, LoadHubertBark, LoadHubertQuantizerBark]