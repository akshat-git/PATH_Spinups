#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build exaonepath.sif exaonepath.def

# Run the spin script inside the container:
#   apptainer exec exaonepath.sif python exaonepath_spin.py
