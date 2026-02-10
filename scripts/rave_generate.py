#!/usr/bin/env python3
"""Generate audio samples using a trained RAVE model."""

import argparse
from pathlib import Path

import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description="Generate audio with RAVE model")
    parser.add_argument("model", help="Path to .ts model file (or directory to search)")
    parser.add_argument("-i", "--input", required=True, help="Input directory with WAV files")
    parser.add_argument("-o", "--output", default="generated", help="Output directory")
    parser.add_argument("-n", "--count", type=int, default=20, help="Number of files to process")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (-1 for CPU)")
    args = parser.parse_args()

    # Find model
    model_path = Path(args.model)
    if model_path.is_dir():
        ts_files = list(model_path.rglob("*.ts"))
        if not ts_files:
            print(f"No .ts file found in {model_path}")
            return 1
        model_path = ts_files[0]

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = torch.jit.load(str(model_path))

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        model = model.to(device)
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Setup directories
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    # Process files
    wav_files = sorted(input_dir.glob("*.wav"))[:args.count]
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return 1

    print(f"Processing {len(wav_files)} files...")

    for f in wav_files:
        print(f"  {f.name}")
        x, sr = torchaudio.load(f)

        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)

        with torch.no_grad():
            out = model.forward(x[None].to(device))

        out_path = output_dir / f.name
        torchaudio.save(str(out_path), out[0].cpu(), sample_rate=model.sr)

    print(f"Done! {len(wav_files)} files saved to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
