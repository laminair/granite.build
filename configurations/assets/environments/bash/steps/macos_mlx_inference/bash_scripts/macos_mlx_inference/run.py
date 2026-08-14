#!/usr/bin/env python3
"""Quantize a HF causal-LM checkpoint to MLX and serve it, as a granite.build step.

Runs locally in the gbserver standalone `bash` environment (Apple Silicon +
Metal only). Parameters come entirely from env vars, set via build.yaml's
`config.bash.env`:
    HF_PATH, MLX_PATH, Q_BITS, Q_GROUP_SIZE, HOST, PORT
Each falls back to the same default as the standalone CLI (see
../../src/mlx_inference/{quantize,serve}.py).
"""

import os
import sys
from pathlib import Path

# Reuse the standalone package's logic rather than duplicating it: this file
# lives at steps/macos_mlx_inference/bash_scripts/macos_mlx_inference/run.py,
# so src/ is two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mlx_inference import quantize, serve  # noqa: E402

ARTIFACT_ID = "mlx_model"


def main() -> None:
    hf_path = os.environ.get("HF_PATH", quantize.DEFAULT_HF_PATH)
    mlx_path = Path(os.environ.get("MLX_PATH", str(quantize.DEFAULT_MLX_PATH)))
    q_bits = int(os.environ.get("Q_BITS", quantize.DEFAULT_Q_BITS))
    q_group_size = int(os.environ.get("Q_GROUP_SIZE", quantize.DEFAULT_Q_GROUP_SIZE))
    host = os.environ.get("HOST", serve.DEFAULT_HOST)
    port = int(os.environ.get("PORT", serve.DEFAULT_PORT))

    quantize.quantize(
        hf_path=hf_path,
        mlx_path=mlx_path,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )

    # Register the quantized model directory as an output before blocking in
    # serve() — the log monitor's regex only needs to see this line once.
    print(f"LLMB_ARTIFACT_ID:{ARTIFACT_ID} LLMB_ARTIFACT_PATH:{mlx_path.resolve()}")

    serve.serve(mlx_path=mlx_path, host=host, port=port)


if __name__ == "__main__":
    main()
