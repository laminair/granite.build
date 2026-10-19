# Copyright 2026 Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Shared GPU base image for granite.build steps. See README.md for what this
# is, when to use it, and the residual caveats (pip can shadow the toolkit;
# BlueVela's pyxis/enroot runtime bypasses this image's ENTRYPOINT entirely,
# so the `ldconfig` refresh it needs lives in each step's own run script, not
# here) that this image does NOT solve by itself.
#
# Pinned to CUDA 12.9.1 to match vLLM 0.27.1's prebuilt `+cu129` wheels (the
# version currently used by granite-evals) -- not NGC's latest tag (13.4.2
# as of writing). Bumping both together is a deliberate, separately-tested
# follow-up; see README.md.
ARG CUDA_IMAGE_TAG=12.9.1-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE_TAG}

ARG DEBIAN_FRONTEND=noninteractive
# The default a bare `uv run`/`uv sync`/`uv venv` resolves to (no --python
# flag); SECONDARY is installed alongside it and reachable via
# `--python 3.13`. See README.md's "Two Python versions, uv-managed" section.
ARG UV_PYTHON_DEFAULT=3.14
ARG UV_PYTHON_SECONDARY=3.13

# build-essential: for a consuming step's own `uv sync` building a C
# extension from source (same rationale bfcl-eval's own Dockerfile gives for
# its build-essential install). No deadsnakes/apt Python here -- uv manages
# Python itself below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv is the default manager for both Python itself and dependencies in this
# image -- not apt/deadsnakes, not system pip. Installed interpreters live
# under UV_PYTHON_INSTALL_DIR, pinned here rather than left at uv's own
# default (~/.local/share/uv/python) so the path stays stable regardless of
# which user/HOME a later consumer layer runs as.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install ${UV_PYTHON_DEFAULT} ${UV_PYTHON_SECONDARY} \
    && uv python pin ${UV_PYTHON_DEFAULT} --no-workspace

# Deliberately NOT symlinking a bare `python3`/`python` to either of these
# (no update-alternatives, unlike the old deadsnakes setup this replaced).
# Every current consumer (e.g. steps/bfcl-eval/skypilot/Dockerfile) runs its
# own `uv sync`, which creates its own `.venv` and prepends `.venv/bin` onto
# PATH -- a bare system-level `python3` was never relied upon. See README.md.

# CUDA toolkit env vars, restated explicitly (sage-style) so the intent is
# self-documenting even though the NGC devel base already sets these itself.
# See README.md's "pip can shadow the toolkit" caveat: a consuming step's own
# dependency install can still land its own CUDA runtime wheels ahead of
# these for anything invoked via `python -m` rather than bare `nvcc`.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
