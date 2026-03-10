"""Build RAVE CLI commands for the TUI training worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

# RAVE v2 requires at least 131072 samples for its multi-scale STFT loss.
RAVE_NUM_SIGNAL = 131072


def _find_source_audio(pool_dir: Path, sr: int, min_duration: float) -> Optional[Path]:
    """Try to find original (unsplit) source audio from pool metadata.

    Returns a directory containing long-enough audio files, or None.
    """
    pool_json = pool_dir / "pool.json"
    if not pool_json.exists():
        return None

    try:
        data = json.loads(pool_json.read_text())
    except Exception:
        return None

    for source in data.get("sources", []):
        src_path = Path(source.get("path", ""))
        # The source path points to the splits dir or file dir.
        # Check the parent for original audio files.
        for candidate in [src_path.parent, src_path]:
            if not candidate.exists():
                continue
            # Look for audio files long enough for RAVE
            for ext in ("*.wav", "*.WAV", "*.flac", "*.mp3"):
                for f in candidate.glob(ext):
                    try:
                        result = subprocess.run(
                            [
                                "ffprobe", "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1",
                                str(f),
                            ],
                            capture_output=True, text=True, timeout=10,
                        )
                        dur = float(result.stdout.strip())
                        if dur >= min_duration:
                            return candidate
                    except Exception:
                        continue
    return None


def prepare_samples(
    samples_dir: Path,
    output_dir: Path,
    pool_dir: Optional[Path] = None,
    sr: int = 44100,
    num_signal: int = RAVE_NUM_SIGNAL,
) -> Path:
    """Find or create audio long enough for RAVE preprocessing.

    Strategy:
    1. If pool samples are already long enough, use them directly.
    2. If pool metadata points to original (unsplit) source audio, use that.
    3. Last resort: concatenate pool samples into one long WAV.

    Returns the directory to pass to ``rave preprocess --input_path``.
    """
    min_duration = num_signal / sr

    # 1. Check if pool samples are long enough as-is
    extensions = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus", "*.aac")
    audio_files: list[Path] = []
    for ext in extensions:
        audio_files.extend(sorted(samples_dir.rglob(ext)))

    if audio_files:
        all_long = True
        for path in audio_files[:20]:
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(path),
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if float(result.stdout.strip()) < min_duration:
                    all_long = False
                    break
            except Exception:
                continue
        if all_long:
            return samples_dir

    # 2. Try original source audio from pool metadata
    if pool_dir is not None:
        source_dir = _find_source_audio(pool_dir, sr, min_duration)
        if source_dir is not None:
            return source_dir

    # 3. Concatenate all samples into one long WAV
    if not audio_files:
        return samples_dir

    concat_dir = output_dir / "concat_audio"
    concat_dir.mkdir(parents=True, exist_ok=True)
    concat_wav = concat_dir / "all_samples.wav"

    if concat_wav.exists():
        return concat_dir

    list_file = concat_dir / "filelist.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for audio in audio_files:
            escaped = str(audio).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-ac", "1", "-ar", str(sr),
            str(concat_wav),
        ],
        check=True,
    )

    return concat_dir


def build_preprocess_cmd(
    rave_cmd: str,
    input_dir: Path,
    output_dir: Path,
    num_signal: int = RAVE_NUM_SIGNAL,
) -> list[str]:
    """Build the rave preprocess command."""
    return [
        rave_cmd, "preprocess",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--num_signal", str(num_signal),
    ]


def build_train_cmd(
    rave_cmd: str,
    config: str,
    data_dir: Path,
    name: str,
    val_every: int = 500,
    max_steps: Optional[int] = None,
    gpu: Optional[int] = None,
    workers: int = 0,
    n_signal: int = RAVE_NUM_SIGNAL,
    ckpt: Optional[Path] = None,
) -> list[str]:
    """Build the rave train command."""
    cmd = [
        rave_cmd, "train",
        "--config", config,
        "--db_path", str(data_dir),
        "--name", name,
        "--n_signal", str(n_signal),
        "--workers", str(workers),
        "--val_every", str(val_every),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    return cmd


def build_export_cmd(
    rave_cmd: str,
    run_dir: str | Path,
    streaming: bool = False,
) -> list[str]:
    """Build the rave export command."""
    cmd = [rave_cmd, "export", "--run", str(run_dir)]
    if streaming:
        cmd.append("--streaming")
    return cmd
