#!/usr/bin/env python3
"""Quantize a HF causal-LM checkpoint to MLX and serve it locally.

Usage:
    uv run steps/macos_mlx_inference/scripts/run.py quantize
    uv run steps/macos_mlx_inference/scripts/run.py serve
    uv run steps/macos_mlx_inference/scripts/run.py all      # quantize (if needed) then serve
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlx_inference import quantize, serve  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize_parser = subparsers.add_parser("quantize", help="Download and quantize the model")
    quantize.add_arguments(quantize_parser)

    serve_parser = subparsers.add_parser("serve", help="Serve the quantized model")
    serve.add_arguments(serve_parser)

    all_parser = subparsers.add_parser("all", help="Quantize (if needed) then serve")
    quantize.add_arguments(all_parser)
    serve.add_arguments(all_parser, include_mlx_path=False)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quantize":
        quantize.run(args)
    elif args.command == "serve":
        serve.run(args)
    elif args.command == "all":
        quantize.run(args)
        serve.run(args)


if __name__ == "__main__":
    main()
