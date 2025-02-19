import contextlib
import re

import funcy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax

from comfy.utils import ProgressBar
from lib.bark.generation import autocast, InferenceContext
from .bark_loader import BarkModelRules
from ...util.categories import category
from ...nodes.basenode import BaseNode

# Constants and functions
SEMANTIC_PAD_TOKEN: int = 10_000
TEXT_ENCODING_OFFSET: int = 10_048
TEXT_PAD_TOKEN: int = 129_595
SEMANTIC_INFER_TOKEN: int = 129_599
SEMANTIC_VOCAB_SIZE = 10_000
SEMANTIC_RATE_HZ: float = 49.9

def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

# Nodes

class PromptEncode(BaseNode):
    name = "bark_text_encode"
    display_name = "bark text encode"

    RETURN_NAMES = ("encoded text",)
    RETURN_TYPES = ("BarkTextEncodings",)
    CATEGORY = category("bark")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_model": ("BarkTextModel",),
                "prompt": ("STRING", {"multiline": True})
            },
            "optional": {
                "semantic_history": ("BarkSemanticHistory",)
            }
        }

    def encode(self, text_model: BarkModelRules, prompt: str, semantic_history = None):
        text = _normalize_whitespace(prompt)

        encoded = np.array(text_model.tokenizer.encode(text, add_special_tokens=False)) + TEXT_ENCODING_OFFSET
        if len(encoded) > 256:
            p = round((len(encoded) - 256) / len(encoded) * 100, 1)
            # logger.warning(f"warning, text too long, lopping of last {p}%")
            encoded = encoded[:256]
        encoded = np.pad(
            encoded,
            (0, 256 - len(encoded)),
            constant_values=TEXT_PAD_TOKEN,
            mode="constant",
        )

        if semantic_history is not None:
            semantic_history = semantic_history.astype(np.int64)
            # lop off if history is too long, pad if needed
            semantic_history = semantic_history[-256:]
            semantic_history = np.pad(
                semantic_history,
                (0, 256 - len(semantic_history)),
                constant_values=SEMANTIC_PAD_TOKEN,
                mode="constant",
            )
        else:
            semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
        x = torch.from_numpy(
            np.hstack([
                encoded, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
            ]).astype(np.int64)
        )[None]

        return (x,)

class GenerateSemantic(BaseNode):
    name = "bark_generate_semantic"
    display_name = "bark generate semantic"

    RETURN_NAMES = ("semantic tokens",)
    RETURN_TYPES = ("BarkSemanticTokens",)
    CATEGORY = category("bark")
    FUNCTION = "generate"

    # temp=0.7,
    #     top_k=None,
    #     top_p=None,
    #     min_eos_p=0.2,
    #     max_gen_duration_s=None,
    #     allow_early_stop=True,
    #     use_kv_caching=False,

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_model": ("BarkTextModel",),
                "encoded_prompt": ("BarkTextEncodings",),
                "temperature": ("FLOAT", {"min": 0.01, "max": 2, "default": 0.7}),
                "top_k": ("INT", {"min": 0, "max": SEMANTIC_VOCAB_SIZE, "default": 0}),
                "top_p": ("FLOAT", {"min": 0, "max": 1, "default": 0}),
                "min_eos_p": ("FLOAT", {"min": 0, "max": 1, "default": 0.2}),
                "allow_early_stop": ("BOOLEAN", {"default": True}),
                "max_gen_duration_s": ("FLOAT", {"min": 0, "max": 15, "default": 0}),
                "use_kv_caching": ("BOOLEAN",)
                # "use_kv_caching": ("BOOLEAN", {"default", False})
            }
        }

    def generate(self, text_model: BarkModelRules, encoded_prompt: torch.Tensor, temperature: float, top_k: int, top_p: float, min_eos_p: float, allow_early_stop: bool, max_gen_duration_s: float, use_kv_caching: bool):
        if top_k == 0:
            top_k = None
        if top_p == 0:
            top_p = None
        if max_gen_duration_s == 0:
            max_gen_duration_s = None

        with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
            device = text_model.patcher.load_device
            text_model.patcher.load(device)
            x = encoded_prompt.to(device)
            model = text_model.model
            n_tot_steps = 768
            pbar = ProgressBar(n_tot_steps)

            tot_generated_duration_s = 0
            kv_cache = None
            for n in range(n_tot_steps):
                if use_kv_caching and kv_cache is not None:
                    x_input = x[:, [-1]]
                else:
                    x_input = x
                logits, kv_cache = model(
                    x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
                )
                relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
                if allow_early_stop:
                    relevant_logits = torch.hstack(
                        (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                    )
                if top_p is not None:
                    # faster to convert to numpy
                    original_device = relevant_logits.device
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(original_device)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temperature, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                if allow_early_stop and (
                        item_next == SEMANTIC_VOCAB_SIZE
                        or (min_eos_p is not None and probs[-1] >= min_eos_p)
                ):
                    # eos found, so break
                    pbar.update_absolute(n_tot_steps)
                    break
                x = torch.cat((x, item_next[None]), dim=1)
                tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
                if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                    pbar.update_absolute(n_tot_steps)
                    break
                if n == n_tot_steps - 1:
                    pbar.update_absolute(n_tot_steps)
                    break
                del logits, relevant_logits, probs, item_next

                pbar.update(1)

            pbar.update_absolute(pbar.total)
            out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1:]
        return (out,)

nodes = [PromptEncode, GenerateSemantic]