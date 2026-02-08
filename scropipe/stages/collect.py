"""Collect stage - pools samples from multiple sources."""

import shutil
from pathlib import Path
from typing import Optional

from .base import Stage, StageResult


class CollectStage(Stage):
    """Stage that collects samples from multiple sources into a unified pool."""

    name = "01-pool"
    description = "Collect samples from splits and includes into a pool"

    def run(
        self,
        split_dirs: Optional[list[Path]] = None,
        include_dirs: Optional[list[Path]] = None,
        symlink: bool = False,
    ) -> StageResult:
        """Run the collect stage.

        Args:
            split_dirs: Directories containing split output (from SplitStage).
            include_dirs: Directories to include directly (existing samples).
            symlink: If True, create symlinks instead of copying files.

        Returns:
            StageResult with success status.
        """
        split_dirs = split_dirs or []
        include_dirs = include_dirs or []

        if not split_dirs and not include_dirs:
            return StageResult(
                success=False,
                message="No input sources provided",
            )

        output_dir = self.ensure_output_dir()
        collected_count = 0
        errors = []

        # Collect from split outputs
        for split_dir in split_dirs:
            split_dir = Path(split_dir)
            if not split_dir.exists():
                errors.append(f"Split directory not found: {split_dir}")
                continue

            # Get the source name from the directory
            source_name = split_dir.name

            # Find all WAV files (may be in subdirectories from scrumpler)
            wav_files = list(split_dir.rglob("*.wav"))

            for i, wav_file in enumerate(wav_files):
                # Create unique name: split_{source}_{index}.wav
                new_name = f"split_{source_name}_{i:04d}.wav"
                dest = output_dir / new_name

                try:
                    if symlink:
                        dest.symlink_to(wav_file.resolve())
                    else:
                        shutil.copy2(wav_file, dest)
                    collected_count += 1
                except Exception as e:
                    errors.append(f"Failed to copy {wav_file}: {e}")

        # Collect from include directories
        for include_dir in include_dirs:
            include_dir = Path(include_dir)
            if not include_dir.exists():
                errors.append(f"Include directory not found: {include_dir}")
                continue

            # Get a short name for the source
            source_name = include_dir.name

            # Find all WAV files
            wav_files = list(include_dir.rglob("*.wav"))

            for wav_file in wav_files:
                # Preserve original filename but add prefix
                # Handle potential name collisions by including relative path
                rel_path = wav_file.relative_to(include_dir)
                if len(rel_path.parts) > 1:
                    # File is in subdirectory, include subdir in name
                    safe_name = "_".join(rel_path.parts).replace(" ", "_")
                else:
                    safe_name = wav_file.name.replace(" ", "_")

                new_name = f"incl_{source_name}_{safe_name}"
                # Ensure .wav extension
                if not new_name.endswith(".wav"):
                    new_name += ".wav"

                dest = output_dir / new_name

                # Handle duplicates by adding a counter
                counter = 1
                base_dest = dest
                while dest.exists():
                    stem = base_dest.stem
                    dest = output_dir / f"{stem}_{counter}.wav"
                    counter += 1

                try:
                    if symlink:
                        dest.symlink_to(wav_file.resolve())
                    else:
                        shutil.copy2(wav_file, dest)
                    collected_count += 1
                except Exception as e:
                    errors.append(f"Failed to copy {wav_file}: {e}")

        if collected_count == 0:
            return StageResult(
                success=False,
                output_dir=output_dir,
                message="No samples collected",
                details={"errors": errors},
            )

        if errors:
            self.log(f"[yellow]Warnings: {len(errors)} errors during collection[/yellow]")
            for err in errors[:5]:  # Show first 5
                self.log(f"  [dim]{err}[/dim]")

        self.log_success(f"Collected {collected_count} samples into pool")

        return StageResult(
            success=True,
            output_dir=output_dir,
            message=f"Collected {collected_count} samples",
            details={
                "sample_count": collected_count,
                "from_splits": len(split_dirs),
                "from_includes": len(include_dirs),
                "errors": errors,
            },
        )
