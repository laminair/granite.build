"""Serve a local MLX model behind an OpenAI-compatible HTTP endpoint."""

import argparse
import subprocess
import sys
from pathlib import Path

from mlx_inference.quantize import DEFAULT_MLX_PATH, is_already_converted

DEFAULT_HOST = "127.0.0.1"
# 8080 collides with gbserver's own default port when both run on the same
# machine; the granite.build recipe overrides this via config.bash.env, but
# keep the standalone-CLI default off of it too.
DEFAULT_PORT = 8085


def serve(
    mlx_path: Path = DEFAULT_MLX_PATH,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    mlx_path = Path(mlx_path)
    if not is_already_converted(mlx_path):
        sys.exit(f"No MLX model found at {mlx_path}. Run the 'quantize' "
                 f"subcommand first.")

    # Resolve against the running interpreter's own bin/, not PATH: the bash
    # step execs this script's venv python directly without ever prepending
    # that venv's bin/ to PATH, so a bare "mlx_lm.server" can resolve to
    # whatever else is reachable on the inherited PATH instead.
    server_bin = str(Path(sys.executable).parent / "mlx_lm.server")
    cmd = [
        server_bin,
        "--model", str(mlx_path),
        "--host", host,
        "--port", str(port),
    ]
    subprocess.run(cmd, check=True)


def add_arguments(parser: argparse.ArgumentParser, include_mlx_path: bool = True) -> None:
    if include_mlx_path:
        parser.add_argument("--mlx-path", default=str(DEFAULT_MLX_PATH),
                             help=f"Local quantized model directory "
                                  f"(default: {DEFAULT_MLX_PATH})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                         help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                         help=f"Server port (default: {DEFAULT_PORT})")


def run(args: argparse.Namespace) -> None:
    serve(mlx_path=Path(args.mlx_path), host=args.host, port=args.port)
