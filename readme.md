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
  * HuBert vectorizer (with output layer option, default 9, main voice quantizer was trained for layer 9)
  * HuBert quantizer for creating npz from audio https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
* Encodec
  * Load models
  * Encode audio
  * Decode audio
  * Coarse-ify codebooks (for creating custom npz files for Bark)
