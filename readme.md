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
  * Load/save speaker npz, put speaker .npz files in `models/bark/speakers`
  * HuBert model loader
  * HuBert quantizer for creating npz from audio https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
* Encodec
  * Load models
  * Encode audio
  * Decode audio

# TODO
* Bark
  * Add an option to select the final layer for the hubert quantizer (similar to clip skip, but for hubert, so let's call it hubert skip)
  * Coarse-ify codebooks for creating custom npz files
  * Direct way to support long-form generations