"""Quantize a Hugging Face causal-LM checkpoint to MLX format."""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_HF_PATH = "ibm-granite/granite-4.2-30b"
DEFAULT_MLX_PATH = Path("models/granite-4.2-30b-4bit")
DEFAULT_Q_BITS = 4
DEFAULT_Q_GROUP_SIZE = 64


def is_already_converted(mlx_path: Path) -> bool:
    return (mlx_path / "config.json").exists()


def quantize(
    hf_path: str = DEFAULT_HF_PATH,
    mlx_path: Path = DEFAULT_MLX_PATH,
    q_bits: int = DEFAULT_Q_BITS,
    q_group_size: int = DEFAULT_Q_GROUP_SIZE,
    force: bool = False,
) -> None:
    mlx_path = Path(mlx_path)
    if is_already_converted(mlx_path) and not force:
        print(f"Found existing MLX model at {mlx_path}, skipping quantization "
              f"(use --force to re-run).")
        return

    print(f"Converting {hf_path} to {q_bits}-bit MLX at {mlx_path}.")
    print("If the source model isn't already in the local HF cache, this "
          "downloads the full-precision weights first (tens of GB) then "
          "writes the quantized output — expect this to take a while and "
          "use significant temporary disk space.")

    mlx_path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve against the running interpreter's own bin/, not PATH: the bash
    # step execs this script's venv python directly without ever prepending
    # that venv's bin/ to PATH, so a bare "mlx_lm.convert" can resolve to
    # whatever else is reachable on the inherited PATH instead.
    convert_bin = str(Path(sys.executable).parent / "mlx_lm.convert")
    cmd = [
        convert_bin,
        "--hf-path", hf_path,
        "--mlx-path", str(mlx_path),
        "-q",
        "--q-bits", str(q_bits),
        "--q-group-size", str(q_group_size),
    ]
    subprocess.run(cmd, check=True)
    print(f"Quantized model written to {mlx_path}.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-path", default=DEFAULT_HF_PATH,
                         help=f"Source HF model (default: {DEFAULT_HF_PATH})")
    parser.add_argument("--mlx-path", default=str(DEFAULT_MLX_PATH),
                         help=f"Local quantized model directory "
                              f"(default: {DEFAULT_MLX_PATH})")
    parser.add_argument("--q-bits", type=int, default=DEFAULT_Q_BITS,
                         help=f"Quantization bit-width (default: {DEFAULT_Q_BITS})")
    parser.add_argument("--q-group-size", type=int, default=DEFAULT_Q_GROUP_SIZE,
                         help=f"Quantization group size (default: {DEFAULT_Q_GROUP_SIZE})")
    parser.add_argument("--force", action="store_true",
                         help="Re-quantize even if the output directory already exists")


def run(args: argparse.Namespace) -> None:
    quantize(
        hf_path=args.hf_path,
        mlx_path=Path(args.mlx_path),
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        force=args.force,
    )
