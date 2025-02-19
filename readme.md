# Comfyui audio nodes
A collection of ComfyUI nodes related to audio.

Current nodes:
* Bark
  * Load models
  * Encode prompt
  * Generate
    1. Semantic
    2. Coarse
    3. Fine
* Encodec
  * Load models
  * Encode audio
  * Decode audio

# TODO
* Bark
  * Load/save speaker npz
    * Use model folder (`models/bark/speakers`) for loading/saving voices
  * Hubert quantizer for creating npz from audio https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
  * Coarse-ify codebooks for creating custom npz files
  * Direct way to support
* Other
  * HuBert model loader
  * Concat audio