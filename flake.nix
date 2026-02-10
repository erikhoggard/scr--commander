{
  description = "Scropipe - Audio pipeline orchestrator for Scrumpler and Scronchler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Tool dependencies - these would be GitHub URLs in production
    # scrumpler.url = "github:erikhoggard/scrumpler";
    # scronchler.url = "github:erikhoggard/scronchler";

    # For local development, use git+file inputs (avoids path resolution issues in Nix 2.31+)
    scrumpler.url = "git+file:///home/erik/dev/scrumpler";
    scronchler.url = "git+file:///home/erik/dev/scronchler";
  };

  outputs = { self, nixpkgs, flake-utils, scrumpler, scronchler }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Get the tools from their flakes
        scrumplerPkg = scrumpler.packages.${system}.default;
        scronchlerPkg = scronchler.packages.${system}.default;

        # Python environment with dependencies
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          typer
          rich
          tomli
          # Dev dependencies
          pytest
          black
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
        scropipe = pkgs.python311Packages.buildPythonApplication {
          pname = "scropipe";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with pkgs.python311Packages; [
            hatchling
          ];

          propagatedBuildInputs = with pkgs.python311Packages; [
            typer
            rich
            tomli
          ];

          # Wrap with tool binaries in PATH
          makeWrapperArgs = [
            "--prefix" "PATH" ":" "${scrumplerPkg}/bin"
            "--prefix" "PATH" ":" "${scronchlerPkg}/bin"
          ];

          meta = with pkgs.lib; {
            description = "Audio pipeline orchestrator for Scrumpler and Scronchler";
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
              scrumplerPkg
              scronchlerPkg
              pkgs.ffmpeg  # Required by RAVE for audio processing
              (pkgs.writeShellScriptBin "scropipe" ''
                exec ${pythonEnv}/bin/python -m scropipe.cli "$@"
              '')
            ];

            # Set library path for pip-installed packages
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

            shellHook = ''
              echo "Scropipe development environment"
              echo ""

              export PYTHONPATH="$PWD:$PYTHONPATH"

              # Setup RAVE in a local venv with ROCm GPU support
              RAVE_VENV="$PWD/.rave-venv"
              if [ ! -d "$RAVE_VENV" ]; then
                echo "Setting up RAVE environment with ROCm GPU support (first time only)..."
                python -m venv "$RAVE_VENV"
                "$RAVE_VENV/bin/pip" install --quiet --upgrade pip
                # Install PyTorch with ROCm 6.2 for AMD GPU support
                "$RAVE_VENV/bin/pip" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
                "$RAVE_VENV/bin/pip" install --quiet acids-rave absl-py
                # RAVE requires scipy 1.10.0 for kaiser function compatibility
                "$RAVE_VENV/bin/pip" install --quiet 'scipy==1.10.0'
                echo "RAVE with ROCm installed!"
              fi
              export PATH="$RAVE_VENV/bin:$PATH"
              # Ensure venv packages take priority over Nix packages
              RAVE_SITE="$RAVE_VENV/lib/python3.11/site-packages"
              export PYTHONPATH="$RAVE_SITE:$PYTHONPATH"

              # AMD GPU (Strix Halo gfx1151) compatibility
              export HSA_OVERRIDE_GFX_VERSION=11.0.0

              echo "Available tools:"
              echo "  - scrumpler (audio splitter)"
              echo "  - scronchler (ML synthesizer)"
              echo "  - scropipe (this package)"
              echo "  - rave (RAVE commands with GPU acceleration)"
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
