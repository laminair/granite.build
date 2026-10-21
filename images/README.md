# images/

Shared base Docker images for granite.build steps. Each subdirectory is a
self-contained image: a `Dockerfile`, a `Makefile` (`make image` /
`make publish-image`), and a `README.md` with usage and caveats.

These are **not** steps (no `step.yaml`, no `space://` URI) — a step's own
Dockerfile builds `FROM` one of these, rather than referencing it as a
granite.build step or environment.

## Images

None currently. `cuda-devel-base/` (CUDA 12.9.1 devel toolkit base) was
removed: its one real candidate consumer, the distillation steps' shared GPU
image, needed a different CUDA minor version (12.8.x, to match
`steps/distill/gpu/`'s vendored `uv.lock`) and a different Python-management
convention (`uv python install`, not apt/deadsnakes) than this directory's
existing pin — rather than fork a second shared-base image here for one
consumer, that image builds directly `FROM nvcr.io/nvidia/cuda:12.8.2-devel-ubuntu22.04`
instead (see `steps/distill/gpu/Dockerfile`). Nothing in this repo ever
built `FROM cuda-devel-base` before its removal.

A future shared base image belongs here if a second, genuinely independent
consumer needs the exact same toolkit/Python pin as an existing step's
image — don't add one speculatively.
