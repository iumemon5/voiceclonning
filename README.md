# Lucy Speech (Inference Only)

Minimal tree for local inference via CLI and Gradio WebUI.

## Requirements
- Python 3.10+.
- CUDA-enabled PyTorch if you want GPU inference (or CPU/MPS/XPU fallbacks).
- Checkpoints (not included here):
  - `checkpoints/lucy5/model.pth`
  - `checkpoints/lucy5/config.json`
  - `checkpoints/lucy5/tokenizer.tiktoken` and related tokenizer files
  - `checkpoints/lucy5/codec.pth`

## Install
```bash
# from repo root
uv sync           # or: pip install -e .
```

## CLI Inference
```bash
python run_cmd.py \
  --text "Hello world" \
  --output output.wav \
  --llama-checkpoint-path checkpoints/lucy5 \
  --decoder-checkpoint-path checkpoints/lucy5/codec.pth \
  --decoder-config-name modded_dac_vq \
  --device cuda   # or cpu/mps/xpu
```
Optional voice cloning:
```bash
python run_cmd.py \
  --text "Hello world" \
  --reference-audio-path path/to/ref.wav \
  --reference-text "the transcript of ref audio"
```

## Gradio WebUI
```bash
python tools/run_webui.py \
  --llama-checkpoint-path checkpoints/lucy5 \
  --decoder-checkpoint-path checkpoints/lucy5/codec.pth \
  --decoder-config-name modded_dac_vq \
  --device cuda   # or cpu/mps/xpu
```
The UI launches on `http://0.0.0.0:8065` by default.

## Notes
- `.project-root` is used by `pyrootutils` to resolve imports; keep it at repo root.
- Adjust paths if you store checkpoints elsewhere.
- For best performance on CUDA, keep `--compile` (default on) and use `--half` for fp16.

## License
Code: Apache-2.0. Model weights: CC-BY-NC-SA-4.0 (check your checkpoint source).
