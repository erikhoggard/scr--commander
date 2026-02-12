{
  description = "Scropipe - Audio pipeline for splitting, collecting, and synthesizing samples";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Python 3.12 environment with all dependencies
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          # CLI
          typer
          rich
          # Splitter (base dependencies)
          numpy
          scipy
          soundfile
          # Dev dependencies
          pytest
          ruff
          hatchling
          pip
        ]);

        # Runtime libraries needed for pip packages (torch, etc.)
        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          zstd  # Required for PyTorch ROCm
          libsndfile
        ];

        # The scropipe package
        scropipe = pkgs.python312Packages.buildPythonApplication {
          pname = "scropipe";
          version = "0.2.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with pkgs.python312Packages; [
            hatchling
          ];

          propagatedBuildInputs = with pkgs.python312Packages; [
            # CLI
            typer
            rich
            # Splitter (base)
            numpy
            scipy
            soundfile
          ];

          meta = with pkgs.lib; {
            description = "Audio pipeline for splitting, collecting, and synthesizing samples";
            license = licenses.mit;
            mainProgram = "scropipe";
          };
        };

      in {
        packages = {
          default = scropipe;
          inherit scropipe;
        };

        devShells = {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.ffmpeg  # Required by RAVE for audio processing
              (pkgs.writeShellScriptBin "scropipe" ''
                exec ${pythonEnv}/bin/python -m scropipe.cli "$@"
              '')
            ];

            # Set library path for pip-installed packages
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

            shellHook = ''
              echo "Scropipe development environment (Python 3.12)"
              echo ""

              export PYTHONPATH="$PWD:$PYTHONPATH"

              # Setup ML dependencies + RAVE in a local venv with ROCm GPU support
              ML_VENV="$PWD/.ml-venv"
              if [ ! -d "$ML_VENV" ]; then
                echo "Setting up ML environment with ROCm GPU support (first time only)..."
                python -m venv "$ML_VENV"
                "$ML_VENV/bin/pip" install --quiet --upgrade pip
                # Install PyTorch with ROCm 6.2 for AMD GPU support
                "$ML_VENV/bin/pip" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
                # Install ML dependencies for scropipe[ml]
                "$ML_VENV/bin/pip" install --quiet librosa
                # Install RAVE
                "$ML_VENV/bin/pip" install --quiet acids-rave absl-py
                # RAVE requires scipy 1.10.0 for kaiser function compatibility
                "$ML_VENV/bin/pip" install --quiet 'scipy==1.10.0'
                echo "ML dependencies + RAVE with ROCm installed!"
              fi
              export PATH="$ML_VENV/bin:$PATH"
              # Ensure venv packages take priority over Nix packages
              ML_SITE="$ML_VENV/lib/python3.12/site-packages"
              export PYTHONPATH="$ML_SITE:$PYTHONPATH"

              # AMD GPU (Strix Halo gfx1151) compatibility
              export HSA_OVERRIDE_GFX_VERSION=11.0.0

              echo "Available commands:"
              echo "  - scropipe      (main pipeline CLI)"
              echo "  - scrumpler     (audio splitter - backwards compat)"
              echo "  - scronchler    (ML synthesizer - backwards compat)"
              echo "  - rave          (RAVE commands with GPU acceleration)"
              echo ""
              echo "Install options:"
              echo "  pip install -e .           # splitting only (lightweight)"
              echo "  pip install -e '.[ml]'     # full ML synthesis"
              echo "  pip install -e '.[ml,dev]' # development"
              echo ""
            '';
          };
        };

        apps.default = {
          type = "app";
          program = "${scropipe}/bin/scropipe";
        };
      }
    );
}
