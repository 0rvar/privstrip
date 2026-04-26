{
  description = "Reference Python environment for openai/privacy-filter (used to validate the Rust port).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python311
            pkgs.uv
          ];

          shellHook = ''
            export UV_PYTHON=${pkgs.python311}/bin/python3.11
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
              echo "Creating uv venv at $UV_PROJECT_ENVIRONMENT"
              uv sync
            fi
            export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
          '';
        };
      });
}
