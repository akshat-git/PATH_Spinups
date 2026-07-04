#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build phikon.sif phikon.def

# Run the spin script inside the container:
#   apptainer exec phikon.sif python phikon_spin.py
