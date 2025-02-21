import contextlib
import re

import funcy
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from encodec.utils import convert_audio
from scipy.special import softmax

from comfy.utils import ProgressBar
from ...lib.bark.generation import autocast, InferenceContext
from .bark_loader import BarkModelRules, BarkHubertModelRules, BarkHubertQuantizerModelRules
from ...util.categories import category
from ...nodes.basenode import BaseNode

# Constants and functions
SEMANTIC_PAD_TOKEN: int = 10_000
TEXT_ENCODING_OFFSET: int = 10_048
TEXT_PAD_TOKEN: int = 129_595
SEMANTIC_INFER_TOKEN: int = 129_599
SEMANTIC_VOCAB_SIZE = 10_000
SEMANTIC_RATE_HZ: float = 49.9

COARSE_RATE_HZ: int = 75
N_COARSE_CODEBOOKS: int = 2
CODEBOOK_SIZE: int = 1024
COARSE_SEMANTIC_PAD_TOKEN: int = 12_048
COARSE_INFER_TOKEN: int = 12_050

N_FINE_CODEBOOKS: int = 8

@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield

def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    if len(arr.shape) == 1:
        arr = arr[None] # Fix
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr

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
                "semantic_history": ("BarkSemanticTokens",)
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
    display_name = "bark generate semantic (1)"

    RETURN_NAMES = ("semantic tokens",)
    RETURN_TYPES = ("BarkSemanticTokens",)
    CATEGORY = category("bark")
    FUNCTION = "generate"

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
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def generate(self, text_model: BarkModelRules, encoded_prompt: torch.Tensor, temperature: float, top_k: int, top_p: float, min_eos_p: float, allow_early_stop: bool, max_gen_duration_s: float, use_kv_caching: bool):
        if top_k == 0:
            top_k = None
        if top_p == 0:
            top_p = None
        if max_gen_duration_s == 0:
            max_gen_duration_s = None

        with _inference_mode():
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

class GenerateCoarse(BaseNode):
    name = "bark_generate_coarse"
    display_name = "bark generate coarse (2)"

    RETURN_NAMES = ("codebooks",)
    RETURN_TYPES = ("EncodecCodeBooks",)
    CATEGORY = category("bark")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coarse_model": ("BarkCoarseModel",),
                "semantic_tokens": ("BarkSemanticTokens",),
                "temperature": ("FLOAT", {"min": 0.01, "max": 2, "default": 0.7}),
                "top_k": ("INT", {"min": 0, "max": SEMANTIC_VOCAB_SIZE, "default": 0}),
                "top_p": ("FLOAT", {"min": 0, "max": 1, "default": 0}),
                "max_coarse_history": ("INT", {"min": 60, "max": 630, "default": 630}),
                "sliding_window_len": ("INT", {"min": 1, "max": 512, "default": 60}),
                "use_kv_caching": ("BOOLEAN",)
            },
            "optional": {
                "semantic_history": ("BarkSemanticTokens",),
                "coarse_history": ("EncodecCodeBooks",)
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def generate(self, coarse_model: BarkModelRules, semantic_tokens: np.ndarray, temperature: float, top_k: int, top_p: float, max_coarse_history: int, sliding_window_len: int, use_kv_caching: bool, semantic_history = None, coarse_history = None):
        if top_k == 0:
            top_k = None
        if top_p == 0:
            top_p = None

        semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        if coarse_history is not None and semantic_history is not None:
            x_semantic_history = semantic_history
            x_coarse_history = coarse_history

            x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
            # trim histories correctly
            n_semantic_hist_provided = np.min(
                [
                    max_semantic_history,
                    len(x_semantic_history) - len(x_semantic_history) % 2,
                    int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
                ]
            )
            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
            x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
            x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)

            x_coarse_history = x_coarse_history[:-2]
        else:
            x_semantic_history = np.array([], dtype=np.int32)
            x_coarse_history = np.array([], dtype=np.int32)

        device = coarse_model.patcher.load_device
        coarse_model.patcher.load(device)
        model = coarse_model.model

        n_steps = int(
            round(
                np.floor(len(semantic_tokens) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
                * N_COARSE_CODEBOOKS
            )
        )
        assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
        x_semantic = np.hstack([x_semantic_history, semantic_tokens]).astype(np.int32)
        x_coarse = x_coarse_history.astype(np.int32)
        base_semantic_idx = len(x_semantic_history)
        with _inference_mode():
            x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
            x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
            n_window_steps = int(np.ceil(n_steps / sliding_window_len))
            n_step = 0
            pbar = ProgressBar(n_window_steps)
            for _ in range(n_window_steps):
                semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
                # pad from right side
                x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]):]
                x_in = x_in[:, :256]
                x_in = F.pad(
                    x_in,
                    (0, 256 - x_in.shape[-1]),
                    "constant",
                    COARSE_SEMANTIC_PAD_TOKEN,
                )
                x_in = torch.hstack(
                    [
                        x_in,
                        torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                        x_coarse_in[:, -max_coarse_history:],
                    ]
                )
                kv_cache = None
                for _ in range(sliding_window_len):
                    if n_step >= n_steps:
                        continue
                    is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                    if use_kv_caching and kv_cache is not None:
                        x_input = x_in[:, [-1]]
                    else:
                        x_input = x_in

                    logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                    logit_start_idx = (
                            SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                    )
                    logit_end_idx = (
                            SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                    )
                    relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
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
                    item_next += logit_start_idx
                    x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                    x_in = torch.cat((x_in, item_next[None]), dim=1)
                    del logits, relevant_logits, probs, item_next
                    n_step += 1
                del x_in
                pbar.update(1)
            del x_semantic_in

        gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history):]
        del x_coarse_in
        assert len(gen_coarse_arr) == n_steps
        gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        for n in range(1, N_COARSE_CODEBOOKS):
            gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
        return gen_coarse_audio_arr

class GenerateFine(BaseNode):
    name = "bark_generate_fine"
    display_name = "bark generate fine (3)"

    RETURN_NAMES = ("codebooks",)
    RETURN_TYPES = ("EncodecCodeBooks",)
    CATEGORY = category("bark")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fine_model": ("BarkFineModel",),
                "coarse_codebooks": ("EncodecCodeBooks",),
                "temperature": ("FLOAT", {"min": 0.01, "max": 2, "default": 0.5})
            },
            "optional": {
                "fine_history": ("EncodecCodeBooks",)
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def generate(self, fine_model: BarkModelRules, coarse_codebooks: numpy.ndarray, temperature: float, fine_history = None):
        if len(coarse_codebooks.shape) == 1:
            coarse_codebooks = np.expand_dims(coarse_codebooks, 0)
        # assert (
        #         isinstance(coarse_codebooks, np.ndarray)
        #         and len(coarse_codebooks.shape) == 2
        #         and 1 <= coarse_codebooks.shape[0] <= N_FINE_CODEBOOKS - 1
        #         and coarse_codebooks.shape[1] > 0
        #         and coarse_codebooks.min() >= 0
        #         and coarse_codebooks.max() <= CODEBOOK_SIZE - 1
        # ), f"{isinstance(coarse_codebooks, np.ndarray)}, {len(coarse_codebooks.shape)}, }"
        if fine_history is not None:
            x_fine_history = fine_history
            assert (
                    isinstance(x_fine_history, np.ndarray)
                    and len(x_fine_history.shape) == 2
                    and x_fine_history.shape[0] == N_FINE_CODEBOOKS
                    and x_fine_history.shape[1] >= 0
                    and x_fine_history.min() >= 0
                    and x_fine_history.max() <= CODEBOOK_SIZE - 1
            )
        else:
            x_fine_history = None

        n_coarse = coarse_codebooks.shape[0]

        device = fine_model.patcher.load_device
        fine_model.patcher.load(device)
        model = fine_model.model


        # make input arr
        in_arr = np.vstack(
            [
                coarse_codebooks,
                np.zeros((N_FINE_CODEBOOKS - n_coarse, coarse_codebooks.shape[1]))
                + CODEBOOK_SIZE,  # padding
            ]
        ).astype(np.int32)
        # prepend history if available (max 512)
        if x_fine_history is not None:
            x_fine_history = x_fine_history.astype(np.int32)
            in_arr = np.hstack(
                [
                    x_fine_history[:, -512:].astype(np.int32),
                    in_arr,
                ]
            )
            n_history = x_fine_history[:, -512:].shape[1]
        else:
            n_history = 0
        n_remove_from_end = 0


        # need to pad if too short (since non-causal model)
        if in_arr.shape[1] < 1024:
            n_remove_from_end = 1024 - in_arr.shape[1]
            in_arr = np.hstack(
                [
                    in_arr,
                    np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
                ]
            )
        # we can be lazy about fractional loop and just keep overwriting codebooks
        n_loops = np.max([0, int(np.ceil((coarse_codebooks.shape[1] - (1024 - n_history)) / 512))]) + 1

        with _inference_mode():
            in_arr = torch.tensor(in_arr.T).to(device)
            for n in range(n_loops):
                start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
                start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
                rel_start_fill_idx = start_fill_idx - start_idx
                in_buffer = in_arr[start_idx: start_idx + 1024, :][None]
                for nn in range(n_coarse, N_FINE_CODEBOOKS):
                    logits = model(nn, in_buffer)
                    if temperature is None:
                        relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                        codebook_preds = torch.argmax(relevant_logits, -1)
                    else:
                        relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temperature
                        probs = F.softmax(relevant_logits, dim=-1)
                        codebook_preds = torch.multinomial(
                            probs[rel_start_fill_idx:1024], num_samples=1
                        ).reshape(-1)
                    codebook_preds = codebook_preds.to(torch.int32)
                    in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                    del logits, codebook_preds
                # transfer over info into model_in and convert to numpy
                for nn in range(n_coarse, N_FINE_CODEBOOKS):
                    in_arr[
                    start_fill_idx: start_fill_idx + (1024 - rel_start_fill_idx), nn
                    ] = in_buffer[0, rel_start_fill_idx:, nn]
                del in_buffer
            gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
            del in_arr

        gen_fine_arr = gen_fine_arr[:, n_history:]
        if n_remove_from_end > 0:
            gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
        assert gen_fine_arr.shape[-1] == coarse_codebooks.shape[-1]

        return (gen_fine_arr,)

class HubertEncode(BaseNode):
    name = "bark_hubert_encode"
    display_name = "HuBERT vectorize audio"

    RETURN_NAMES = ("vectors",)
    RETURN_TYPES = ("HuBERTVectors",)
    CATEGORY = category("bark/cloning")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hubert_model": ("BarkHuBERTModel",),
                "audio": ("AUDIO",)
            }
        }

    def encode(self, hubert_model: BarkHubertModelRules, audio):
        device = hubert_model.patcher.load_device
        hubert_model.patcher.load(device)
        model = hubert_model.model

        wav: torch.Tensor = audio["waveform"]
        sr: int = audio["sample_rate"]
        wav = convert_audio(wav, sr, model.target_sample_hz, 1)
        wav = wav.to(device)[0] # shape was [1, 1, 1, 179840]

        vecs = model.forward(wav, input_sample_hz=sr)

        return (vecs,)

class HubertQuantize(BaseNode):
    name = "bark_hubert_quantize"
    display_name = "quantize HuBERT vectors to bark semantic tokens"

    RETURN_NAMES = ("semantic_tokens",)
    RETURN_TYPES = ("BarkSemanticTokens",)
    CATEGORY = category("bark/cloning")
    FUNCTION = "quantize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quantizer_model": ("BarkHuBERTQuantizerModel",),
                "vectors": ("HuBERTVectors",)
            }
        }

    def quantize(self, quantizer_model: BarkHubertQuantizerModelRules, vectors: torch.Tensor):
        device = quantizer_model.patcher.load_device
        quantizer_model.patcher.load(device)
        model = quantizer_model.model

        vectors = vectors.to(device)
        tokens = model.get_token(vectors)

        return (tokens,)

class EncodecCourseify(BaseNode):
    name = "bark_encodec_coarseify"
    display_name = "Course-ify encodec codebooks"

    RETURN_NAMES = ("coarse_codebooks",)
    RETURN_TYPES = ("EncodecCodeBooks",)
    CATEGORY = category("bark")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fine_codebooks": ("EncodecCodeBooks",)
            }
        }

    def encode(self, fine_codebooks):
        return (fine_codebooks[:2, :],)

nodes = [PromptEncode, GenerateSemantic, GenerateCoarse, GenerateFine, HubertEncode, HubertQuantize, EncodecCourseify]