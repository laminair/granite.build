# Docker packaging — deferred

MLX requires Apple Silicon + Metal and cannot run inside a Linux container,
so this step is bare-metal only for now. Revisit once there's a concrete
plan for running the MLX runtime itself outside of macOS (or for wrapping
just the OpenAI-compatible HTTP layer in a container that talks to a
bare-metal MLX process).
