from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from encodec.model import EncodecModel
from encodec.utils import convert_audio

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
    CATEGORY = category("general/encodec")
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

class EncodecDecode(BaseNode):
    name = "encodec_decode"
    display_name = "Decode encodec codebooks"

    RETURN_NAMES = ("audio",)
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = category("general/encodec")
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encodec_model": ("EncodecModel",),
                "codebooks": ("EncodecCodeBooks",)
            }
        }

    def decode(self, encodec_model: EncodecModelRules, codebooks: np.ndarray):
        device = encodec_model.patcher.load_device
        encodec_model.patcher.load(device)
        model = encodec_model.model

        arr = torch.from_numpy(codebooks)[None]
        if len(arr.shape) == 2:
            arr = arr[None] # Unsqueeze again
        arr = arr.to(device)
        arr = arr.transpose(0, 1)
        emb = model.quantizer.decode(arr)
        out = model.decoder(emb)
        audio_arr = out.detach().squeeze()
        del arr, emb, out

        audio_arr = audio_arr[None]
        if len(audio_arr.shape) == 2:
            audio_arr = audio_arr[None]

        return ({"waveform": audio_arr, "sample_rate": encodec_model.model.sample_rate},)

class EncodecEncode(BaseNode):
    name = "encodec_encode"
    display_name = "Encode encodec codebooks"

    RETURN_NAMES = ("codebooks",)
    RETURN_TYPES = ("EncodecCodeBooks",)
    CATEGORY = category("general/encodec")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encodec_model": ("EncodecModel",),
                "audio": ("AUDIO",)
            }
        }

    def encode(self, encodec_model: EncodecModelRules, audio: dict):
        device = encodec_model.patcher.load_device
        encodec_model.patcher.load(device)
        model = encodec_model.model

        wav: torch.Tensor = audio["waveform"][0]
        sr: int = audio["sample_rate"]
        wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)
        wav = wav.to(device)

        with torch.no_grad():
            codebooks = model.encode(wav)
        codebooks = torch.cat([encoded[0] for encoded in codebooks], dim=-1).squeeze()
        codebooks = codebooks.cpu().numpy()

        return (codebooks,)


nodes = [EncodecLoader, EncodecDecode, EncodecEncode]