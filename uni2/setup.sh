#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build uni2.sif uni2.def

# Run the spin script inside the container:
#   apptainer exec uni2.sif python uni2_spin.py
