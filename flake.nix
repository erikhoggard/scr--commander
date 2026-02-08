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
              (pkgs.writeShellScriptBin "scropipe" ''
                exec ${pythonEnv}/bin/python -m scropipe.cli "$@"
              '')
            ];

            shellHook = ''
              echo "Scropipe development environment"
              echo ""
              echo "Available tools:"
              echo "  - scrumpler (audio splitter)"
              echo "  - scronchler (ML synthesizer)"
              echo "  - scropipe (this package)"
              echo ""

              export PYTHONPATH="$PWD:$PYTHONPATH"
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
