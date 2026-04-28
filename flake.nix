{
  description = "privstrip — Rust port of openai/privacy-filter, plus Python reference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          name = "privstrip";
          packages = with pkgs; [
            # Rust toolchain (use rustup if you need a specific channel; this gives you stable).
            cargo
            rustc
            rust-analyzer
            clippy
            rustfmt

            # Python reference — uv manages the actual interpreter & deps via python-ref/uv.lock.
            uv
            python311

            # Validation harness.
            bun

            # Profiler used to validate the optimization pass against the
            # bf16/f32 baseline (CPU + Metal forward).
            samply
          ];

          shellHook = ''
            export PRIVSTRIP_REPO_ROOT=${toString ./.}
            # Help uv find a CPython 3.11 without re-downloading.
            export UV_PYTHON_PREFERENCE=only-system
          '';
        };
      });
}
