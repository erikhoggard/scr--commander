#!/usr/bin/env python3
"""
Quick batch processor with common presets for experimental sampling
"""

from __future__ import annotations

import sys
from typing import Any

from .processor import SampleProcessor


# Type alias for preset configuration
PresetConfig = dict[str, str | list[str] | dict[str, Any]]

PRESETS: dict[str, PresetConfig] = {
    'quick': {
        'description': 'Quick exploration - all modes with 1.5s grid chunks',
        'modes': ['grid', 'transient', 'texture'],
        'params': {'chunk_length': 1.5}
    },

    'loops': {
        'description': 'Extract 2-bar loops at 120 BPM',
        'modes': ['grid'],
        'params': {'bpm': 120, 'bars': 2}
    },

    'loops_90': {
        'description': '4-bar loops at 90 BPM (hip-hop tempo)',
        'modes': ['grid'],
        'params': {'bpm': 90, 'bars': 4}
    },

    'loops_120': {
        'description': '4-bar loops at 120 BPM (house tempo)',
        'modes': ['grid'],
        'params': {'bpm': 120, 'bars': 4}
    },

    'loops_128': {
        'description': '4-bar loops at 128 BPM (techno tempo)',
        'modes': ['grid'],
        'params': {'bpm': 128, 'bars': 4}
    },

    'loops_140': {
        'description': '4-bar loops at 140 BPM (dubstep/grime tempo)',
        'modes': ['grid'],
        'params': {'bpm': 140, 'bars': 4}
    },

    'loops_160': {
        'description': '2-bar loops at 160 BPM (drum & bass tempo)',
        'modes': ['grid'],
        'params': {'bpm': 160, 'bars': 2}
    },

    'loops_174': {
        'description': '2-bar loops at 174 BPM (jungle tempo)',
        'modes': ['grid'],
        'params': {'bpm': 174, 'bars': 2}
    },

    'one_bar': {
        'description': '1-bar loops at 120 BPM',
        'modes': ['grid'],
        'params': {'bpm': 120, 'bars': 1}
    },

    'eight_bar': {
        'description': '8-bar loops at 120 BPM',
        'modes': ['grid'],
        'params': {'bpm': 120, 'bars': 8}
    },

    'drums': {
        'description': 'Sensitive transient detection for drum hits',
        'modes': ['transient'],
        'params': {'delta': 0.05, 'max_length': 2.0}
    },

    'percussion': {
        'description': 'Less sensitive, for cleaner percussion',
        'modes': ['transient'],
        'params': {'delta': 0.1, 'max_length': 1.5}
    },

    'drones': {
        'description': 'Long ambient textures and drones',
        'modes': ['texture'],
        'params': {'min_duration': 3.0, 'max_duration': 60.0,
                   'rms_threshold': 0.1, 'stability_threshold': 0.1}
    },

    'pads': {
        'description': 'Medium-length ambient textures',
        'modes': ['texture'],
        'params': {'min_duration': 1.0, 'max_duration': 15.0,
                   'rms_threshold': 0.08, 'stability_threshold': 0.12}
    },

    'granular': {
        'description': 'Tiny chunks for granular synthesis (250ms)',
        'modes': ['grid'],
        'params': {'chunk_length': 0.25}
    },

    'speech': {
        'description': 'Extract sustained textures, avoid speech',
        'modes': ['texture'],
        'params': {'min_duration': 2.0, 'max_duration': 30.0,
                   'rms_threshold': 0.2, 'stability_threshold': 0.08}
    },

    'all_modes': {
        'description': 'Process with all three modes',
        'modes': ['grid', 'transient', 'texture'],
        'params': {}
    },

    'hits_and_loops': {
        'description': 'Extract both transients and grid chunks',
        'modes': ['grid', 'transient'],
        'params': {'chunk_length': 2.0, 'delta': 0.06}
    },

    'full_extract': {
        'description': 'Extract everything: grid, transients, and textures',
        'modes': ['grid', 'transient', 'texture'],
        'params': {'chunk_length': 1.5}
    },

    'left_channel': {
        'description': 'Process left channel only (all modes)',
        'modes': ['grid', 'transient', 'texture'],
        'params': {'channel': 'left'}
    },

    'right_channel': {
        'description': 'Process right channel only (all modes)',
        'modes': ['grid', 'transient', 'texture'],
        'params': {'channel': 'right'}
    },
}

# Default parameter values (matching sample_processor.py defaults)
DEFAULTS: dict[str, Any] = {
    'input': '_incoming_rips',
    'output': '_chopped_samples',
    'sr': 44100,
    'channel': 'mono',
    'chunk_length': 2.0,
    'bpm': None,
    'bars': 4,
    'delta': 0.07,
    'min_length': 0.05,
    'max_length': 10.0,
    'min_duration': 1.0,
    'max_duration': 30.0,
    'rms_threshold': 0.1,
    'stability_threshold': 0.15,
}


class Args:
    """Simple namespace to hold arguments, mimicking argparse.Namespace."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def print_presets() -> None:
    """Print all available presets."""
    print("\n=== Available Presets ===\n")
    for name, preset in PRESETS.items():
        print(f"  {name:16} - {preset['description']}")
    print()


def run_preset(preset_name: str, file_or_batch_arg: str | None = None) -> int:
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print_presets()
        return 1

    preset = PRESETS[preset_name]
    modes = preset['modes']

    # Build args from defaults + preset params
    args_dict = DEFAULTS.copy()
    args_dict.update(preset['params'])

    # Calculate chunk_length from BPM if specified
    if args_dict.get('bpm'):
        args_dict['chunk_length'] = (60.0 / args_dict['bpm']) * args_dict['bars'] * 4

    args = Args(**args_dict)

    print(f"\n{'=' * 60}")
    print(f"Running preset: {preset_name}")
    print(f"Description: {preset['description']}")

    processor = SampleProcessor(args.input, args.output, sr=args.sr)

    if file_or_batch_arg and file_or_batch_arg != '--batch':
        print(f"Processing: {file_or_batch_arg}")
        print(f"{'=' * 60}\n")
        processor.process_single_file(file_or_batch_arg, modes, args)
    else:
        print(f"Processing: All files in {args.input}/")
        print(f"{'=' * 60}\n")
        processor.process_directory(modes, args)

    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print("\nUsage: scrumpler-batch <preset> [file|--batch]")
        print_presets()
        print("Examples:")
        print("  scrumpler-batch quick                 # Process all files")
        print("  scrumpler-batch quick --batch         # Process all files (explicit)")
        print("  scrumpler-batch drones abc.wav        # Process specific file")
        print("  scrumpler-batch drums song.wav        # Process specific file")
        sys.exit(1)

    preset_name = sys.argv[1]

    if preset_name in ['--help', '-h', 'help']:
        print_presets()
        sys.exit(0)

    file_or_batch = sys.argv[2] if len(sys.argv) > 2 else None

    sys.exit(run_preset(preset_name, file_or_batch))


if __name__ == '__main__':
    main()
