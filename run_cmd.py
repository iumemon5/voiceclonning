import argparse
import os
from pathlib import Path
import soundfile as sf
import torch
import pyrootutils

# Setup root path
# Locate the project root relative to this script
# If this script is in the root, .project-root should be here.
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from lucy_speech.inference_engine import TTSInferenceEngine
from lucy_speech.models.dac.inference import load_model as load_decoder_model
from lucy_speech.models.text2semantic.inference import launch_thread_safe_queue
from lucy_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

def parse_args():
    parser = argparse.ArgumentParser(description="Run Lucy Speech inference from CLI")
    parser.add_argument("--text", type=str, required=True, help="Text to generate speech from")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--llama-checkpoint-path", type=Path, default="checkpoints/lucy5")
    parser.add_argument("--decoder-checkpoint-path", type=Path, default="checkpoints/lucy5/codec.pth")
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", default=True, action="store_true", help="Use half precision (fp16)")
    parser.add_argument("--compile", default=True, action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep models loaded and enter multiple prompts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed (set to a fixed value for reproducible outputs)",
    )
    parser.add_argument(
        "--reference-id",
        type=str,
        help="Reference ID from the 'references/<id>' folder (uses all samples there)",
    )
    parser.add_argument(
        "--use-memory-cache",
        choices=["on", "off"],
        default="on",
        help="Reuse encoded references in memory for faster repeat runs",
    )

    # Add reference voice arguments
    parser.add_argument("--reference-audio-path", type=Path, help="Path to the reference audio file (e.g., .wav)")
    parser.add_argument("--reference-text", type=str, help="Text corresponding to the reference audio")

    return parser.parse_args()

def main():
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    if torch.backends.mps.is_available():
        args.device = "mps"
        print("mps is available, running on mps.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        print("XPU is available, running on XPU.")
    elif not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA is not available, running on CPU.")
        args.device = "cpu"

    print("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    print("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    print("Initializing inference engine...")
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    references = []
    reference_id = args.reference_id

    if reference_id:
        print(f"Using stored reference ID: '{reference_id}' (from references/{reference_id})")
    elif args.reference_audio_path and args.reference_text:
        if not args.reference_audio_path.exists():
            print(f"Error: Reference audio file not found at {args.reference_audio_path}")
            return

        print(f"Loading reference audio from {args.reference_audio_path}...")
        with open(args.reference_audio_path, "rb") as f:
            reference_audio_bytes = f.read()

        references.append(ServeReferenceAudio(audio=reference_audio_bytes, text=args.reference_text))
        print(f"Using reference voice with text: '{args.reference_text}'")
    elif args.reference_audio_path or args.reference_text:
        print("Warning: Provide both --reference-audio-path and --reference-text, or use --reference-id.")


    output_path = Path(args.output)
    out_parent = output_path.parent
    out_stem = output_path.stem
    out_ext = output_path.suffix if output_path.suffix else ".wav"

    def out_file(idx: int) -> Path:
        if idx == 0:
            return output_path
        return out_parent / f"{out_stem}_{idx}{out_ext}"

    def run_once(text_value: str, dest: Path):
        req = ServeTTSRequest(
            text=text_value,
            references=references,
            reference_id=reference_id,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            seed=args.seed,
            format="wav",
            use_memory_cache=args.use_memory_cache,
        )

        print(f"Generating audio for: '{text_value}'")
        for result in engine.inference(req):
            if result.code == "final":
                if isinstance(result.audio, tuple):
                    sr, audio_data = result.audio
                    sf.write(dest, audio_data, sr)
                    print(f"Saved to {dest}")
                else:
                    print("Warning: Final result did not contain tuple audio data.")
                break
            elif result.code == "error":
                print(f"Error during inference: {result.error}")
                break

    idx = 0
    run_once(args.text, out_file(idx))
    idx += 1

    if args.interactive:
        print("Interactive mode enabled. Enter text (empty line to exit).")
        while True:
            try:
                new_text = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not new_text:
                break
            run_once(new_text, out_file(idx))
            idx += 1

if __name__ == "__main__":
    main()
