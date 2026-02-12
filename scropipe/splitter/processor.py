#!/usr/bin/env python3
"""
Audio Sample Processor for Experimental Music Production
Processes audio files with grid chopping, transient detection, and texture gating
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing import Protocol

    class ProcessingArgs(Protocol):
        """Protocol defining expected attributes on args objects."""
        channel: str
        chunk_length: float
        bpm: float | None
        bars: int
        delta: float
        min_length: float
        max_length: float
        min_duration: float
        max_duration: float
        rms_threshold: float
        stability_threshold: float

try:
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import stft
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Please run: pip install soundfile scipy numpy")
    sys.exit(1)

# Type aliases
AudioArray = npt.NDArray[np.floating]

# =============================================================================
# Processing Constants
# =============================================================================

# Minimum segment duration to keep (in seconds) - filters out tiny fragments
MIN_SEGMENT_DURATION_SEC = 0.05

# Default FFT/analysis parameters
DEFAULT_HOP_LENGTH = 512
DEFAULT_FRAME_LENGTH = 2048

# Transient detection parameters
GAUSSIAN_SMOOTHING_SIGMA = 2
ONSET_DEBOUNCE_FRAMES = 10

# Spectral analysis parameters
STABILITY_WINDOW_FRAMES = 20
EPSILON = 1e-8  # Small value to prevent division by zero


class SampleProcessor:
    def __init__(self, input_dir: str | Path | None, output_dir: str | Path, sr: int = 44100) -> None:
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir)
        self.sr = sr

    def _generate_params_string(self, args: ProcessingArgs, mode: str) -> str:
        """Generate a string of relevant parameters for a given mode."""
        params = []
        if mode == 'grid':
            if args.bpm:
                params.append(f"bpm{args.bpm}")
                if args.bars:
                    params.append(f"bars{args.bars}")
            elif args.chunk_length is not None:
                params.append(f"len{args.chunk_length:.2f}s".replace('.', 'p'))
        elif mode == 'transient':
            params.append(f"delta{args.delta:.2f}".replace('.', 'p'))
            params.append(f"min{args.min_length:.2f}s".replace('.', 'p'))
            params.append(f"max{args.max_length:.1f}s".replace('.', 'p'))
        elif mode == 'texture':
            params.append(f"min{args.min_duration:.1f}s".replace('.', 'p'))
            params.append(f"max{args.max_duration:.1f}s".replace('.', 'p'))
            params.append(f"rms{args.rms_threshold:.2f}".replace('.', 'p'))
            params.append(f"stab{args.stability_threshold:.2f}".replace('.', 'p'))

        if args.channel and args.channel != 'mono':
            params.append(args.channel)

        return "_" + "_".join(params) if params else ""

    def process_single_file(self, filepath: str | Path, modes: list[str], args: ProcessingArgs) -> None:
        """Process a single audio file with specified modes."""
        filepath = Path(filepath)
        if not filepath.exists():
            if self.input_dir:
                filepath = self.input_dir / filepath.name
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return
        if filepath.suffix.lower() != '.wav':
            print(f"Error: File must be a WAV file: {filepath}")
            return

        print(f"\n{'=' * 80}")
        print(f"Processing: {filepath.name}")
        print(f"Modes: {', '.join(modes)}")
        print(f"{'=' * 80}")

        y, sr = sf.read(filepath, dtype='float32')

        channel = getattr(args, 'channel', 'mono')
        if y.ndim == 2:
            if channel == 'mono':
                y = np.mean(y, axis=1)
                print(f"Channel: Mono (summed)")
            elif channel == 'left':
                y = y[:, 0]
                print(f"Channel: Left only")
            elif channel == 'right':
                y = y[:, 1]
                print(f"Channel: Right only")

        if sr != self.sr:
            num_samples = int(len(y) * self.sr / sr)
            y = signal.resample(y, num_samples)
            sr = self.sr

        print(f"Duration: {len(y) / sr:.2f}s, Sample rate: {sr}Hz")

        # Sanitize original_file_name for the master folder
        sanitized_stem = filepath.stem.replace('.', '_')

        # Create the master folder: _chopped_samples/original_file_name/
        master_output_folder = self.output_dir / sanitized_stem
        master_output_folder.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            param_string = self._generate_params_string(args, mode)
            mode_subfolder_name = f"{mode}{param_string}"

            # Create the subfolder: _chopped_samples/original_file_name/mode_params/
            mode_output_folder = master_output_folder / mode_subfolder_name
            mode_output_folder.mkdir(exist_ok=True)

            print(f"\n--- Running Mode: {mode.upper()}{param_string} ---")

            if mode == 'grid':
                segments = self._grid_chop(y, sr, args.chunk_length)
            elif mode == 'transient':
                segments = self._transient_split(y, sr, args)
            elif mode == 'texture':
                segments = self._texture_gate(y, sr, args)
            else:
                print(f"Unknown mode: {mode}")
                continue

            # Here, the base_name is the original sanitized file stem, and the mode is passed separately for the filename
            self._save_segments(segments, mode_output_folder, sanitized_stem, mode)

        print(f"\n{'=' * 80}")
        print(f"[OK] Processing complete for: {filepath.name}. Output in: {master_output_folder}")
        print(f"{'=' * 80}\n")

    def process_directory(self, modes: list[str], args: ProcessingArgs) -> None:
        """Process all WAV files in the input directory with specified modes."""
        if not self.input_dir:
            print("Error: No input directory specified")
            return

        wav_files = list(self.input_dir.glob('*.wav'))
        if not wav_files:
            print(f"No WAV files found in {self.input_dir}")
            return

        print(f"\n{'=' * 80}")
        print(f"Found {len(wav_files)} WAV files for batch processing")
        print(f"Modes: {', '.join(modes)}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 80}")

        for filepath in wav_files:
            try:
                self.process_single_file(filepath, modes, args)
            except Exception as e:
                print(f"FATAL Error processing {filepath.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'=' * 80}")
        print(f"[OK] Batch processing complete!")
        print(f"{'=' * 80}\n")

    def _grid_chop(self, y: AudioArray, sr: int, chunk_length: float) -> list[AudioArray]:
        """Chop audio into equal-length chunks."""
        chunk_samples = int(chunk_length * sr)
        if chunk_samples == 0:
            print("Warning: Chunk length is too small, skipping grid chop.")
            return []
        min_samples = int(sr * MIN_SEGMENT_DURATION_SEC)
        segments = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]
        segments = [s for s in segments if len(s) > min_samples]
        print(f"Grid chopped into {len(segments)} segments of ~{chunk_length:.2f}s")
        return segments

    def _transient_split(self, y: AudioArray, sr: int, args: ProcessingArgs) -> list[AudioArray]:
        """Split audio on transient peaks (percussive hits)."""
        hop_length = getattr(args, 'hop_length', DEFAULT_HOP_LENGTH)
        frame_length = hop_length * 2

        num_frames = (len(y) - frame_length) // hop_length + 1
        frames = np.lib.stride_tricks.as_strided(
            y,
            shape=(num_frames, frame_length),
            strides=(y.strides[0] * hop_length, y.strides[0])
        )
        energy = np.sum(frames**2, axis=1)

        energy_smooth = gaussian_filter1d(energy, sigma=GAUSSIAN_SMOOTHING_SIGMA)

        onset_strength = np.diff(energy_smooth, prepend=energy_smooth[0])
        onset_strength[onset_strength < 0] = 0

        if np.max(onset_strength) > 0:
            onset_strength /= np.max(onset_strength)

        delta = getattr(args, 'delta', 0.07)
        wait_frames = getattr(args, 'wait', ONSET_DEBOUNCE_FRAMES)

        onset_frames: list[int] = []
        last_onset = -wait_frames
        for i, strength in enumerate(onset_strength):
            if strength > delta and (i - last_onset) >= wait_frames:
                onset_frames.append(i)
                last_onset = i

        onset_samples = np.array([f * hop_length for f in onset_frames])
        onset_samples = np.concatenate([[0], onset_samples, [len(y)]])

        segments: list[AudioArray] = []
        min_len = int(sr * getattr(args, 'min_length', MIN_SEGMENT_DURATION_SEC))
        max_len = int(sr * getattr(args, 'max_length', 10.0))

        for i in range(len(onset_samples) - 1):
            segment = y[onset_samples[i]:onset_samples[i+1]]
            if min_len <= len(segment) <= max_len:
                segments.append(segment)

        print(f"Detected {len(onset_frames)} transients, extracted {len(segments)} valid segments")
        return segments

    def _texture_gate(self, y: AudioArray, sr: int, args: ProcessingArgs) -> list[AudioArray]:
        """Extract ambient/sustained textures using spectral and energy analysis."""
        frame_length = getattr(args, 'frame_length', DEFAULT_FRAME_LENGTH)
        hop_length = getattr(args, 'hop_length', DEFAULT_HOP_LENGTH)

        f, t, Zxx = stft(y, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)
        mag = np.abs(Zxx)

        rms = np.sqrt(np.mean(mag**2, axis=0))
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + EPSILON)

        freqs = f.reshape(-1, 1)
        centroid = np.sum(freqs * mag, axis=0) / (np.sum(mag, axis=0) + EPSILON)
        centroid_norm = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + EPSILON)

        window = getattr(args, 'stability_window', STABILITY_WINDOW_FRAMES)
        num_frames = mag.shape[1]
        centroid_std = np.array([np.std(centroid_norm[max(0, i-window):i+window]) for i in range(num_frames)])

        is_texture = (
            (rms_norm > getattr(args, 'rms_threshold', 0.1)) &
            (centroid_std < getattr(args, 'stability_threshold', 0.15))
        )

        segments: list[AudioArray] = []
        min_samples = int(getattr(args, 'min_duration', 1.0) * sr)
        max_samples = int(getattr(args, 'max_duration', 30.0) * sr)

        in_region = False
        region_start = 0
        for i, is_tex in enumerate(is_texture):
            sample_idx = i * hop_length
            if is_tex and not in_region:
                region_start = sample_idx
                in_region = True
            elif not is_tex and in_region:
                region = y[region_start:sample_idx]
                if min_samples <= len(region) <= max_samples:
                    segments.append(region)
                in_region = False
        if in_region:
            region = y[region_start:len(y)]
            if min_samples <= len(region) <= max_samples:
                segments.append(region)

        print(f"Extracted {len(segments)} textural segments")
        return segments

    def _save_segments(self, segments: list[AudioArray], output_folder: Path, base_name: str, mode: str) -> None:
        """Save audio segments to files."""
        if not segments:
            print(f"No segments to save for {mode} mode.")
            return

        for i, segment in enumerate(segments):
            # Normalize segment to avoid clipping
            peak = np.max(np.abs(segment))
            if peak > 1.0:
                segment /= peak

            output_path = output_folder / f"{base_name}_{i:04d}.wav"
            sf.write(output_path, segment, self.sr)

        print(f"Saved {len(segments)} segments to: {output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Process audio samples for experimental music production',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process specific file with one mode
  scrumpler abc.wav --grid --chunk-length 2.0

  # Musical grid chopping at specific BPM
  scrumpler abc.wav --grid --bpm 120 --bars 2
  scrumpler abc.wav --grid --bpm 140 --bars 4

  # Process all files in directory with specific modes
  scrumpler --batch --grid --transient --delta 0.1

  # Batch process with musical timing
  scrumpler --batch --grid --bpm 128 --bars 4
        '''
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Specific WAV file to process (e.g., abc.wav or _incoming_rips/abc.wav)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all WAV files in input directory instead of a single file'
    )

    parser.add_argument(
        '-i', '--input',
        default='_incoming_rips',
        help='Input directory containing WAV files (default: _incoming_rips)'
    )

    parser.add_argument(
        '-o', '--output',
        default='_chopped_samples',
        help='Output directory for processed samples (default: _chopped_samples)'
    )

    # Mode selection flags
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Enable grid chopping mode'
    )

    parser.add_argument(
        '--transient',
        action='store_true',
        help='Enable transient detection mode'
    )

    parser.add_argument(
        '--texture',
        action='store_true',
        help='Enable texture gating mode'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Enable all processing modes'
    )

    parser.add_argument(
        '--sr',
        type=int,
        default=44100,
        help='Sample rate for processing (default: 44100)'
    )

    parser.add_argument(
        '--channel',
        choices=['left', 'right', 'mono'],
        default='mono',
        help='Channel selection: left, right, or mono (sum to mono, default)'
    )

    # Grid mode parameters
    parser.add_argument(
        '--chunk-length',
        type=float,
        default=None,
        help='Length of chunks in seconds for grid mode (default: 2.0, overridden by --bpm/--bars)'
    )

    parser.add_argument(
        '--bpm',
        type=float,
        default=None,
        help='BPM for musical grid chopping (use with --bars, overrides --chunk-length)'
    )

    parser.add_argument(
        '--bars',
        type=int,
        default=4,
        help='Number of bars per chunk when using --bpm (default: 4 bars)'
    )

    # Transient mode parameters
    parser.add_argument(
        '--delta',
        type=float,
        default=0.07,
        help='Transient detection sensitivity (default: 0.07, lower = more sensitive)'
    )

    parser.add_argument(
        '--min-length',
        type=float,
        default=0.05,
        help='Minimum segment length in seconds for transient mode (default: 0.05)'
    )

    parser.add_argument(
        '--max-length',
        type=float,
        default=10.0,
        help='Maximum segment length in seconds for transient mode (default: 10.0)'
    )

    # Texture mode parameters
    parser.add_argument(
        '--min-duration',
        type=float,
        default=1.0,
        help='Minimum duration for texture segments in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--max-duration',
        type=float,
        default=30.0,
        help='Maximum duration for texture segments in seconds (default: 30.0)'
    )

    parser.add_argument(
        '--rms-threshold',
        type=float,
        default=0.1,
        help='RMS threshold for texture detection (default: 0.1)'
    )

    parser.add_argument(
        '--stability-threshold',
        type=float,
        default=0.15,
        help='Spectral stability threshold for texture detection (default: 0.15)'
    )

    args = parser.parse_args()

    modes = []
    if args.all:
        modes = ['grid', 'transient', 'texture']
    else:
        if args.grid: modes.append('grid')
        if args.transient: modes.append('transient')
        if args.texture: modes.append('texture')

    if not modes:
        print("Error: No processing mode specified. Use --grid, --transient, --texture, or --all")
        parser.print_help()
        sys.exit(1)

    if not args.batch and not args.file:
        print("Error: Must specify either a file to process or use --batch")
        parser.print_help()
        sys.exit(1)

    processor = SampleProcessor(args.input, args.output, sr=args.sr)

    if args.bpm:
        args.chunk_length = (60.0 / args.bpm) * args.bars * 4
        print(f"\n{'=' * 60}")
        print(f"Musical Grid Mode: {args.bpm} BPM, {args.bars} bars = {args.chunk_length:.3f}s chunks")
        print(f"{'=' * 60}")
    elif args.chunk_length is None:
        args.chunk_length = 2.0

    if args.batch:
        processor.process_directory(modes, args)
    else:
        processor.process_single_file(args.file, modes, args)


if __name__ == '__main__':
    main()
