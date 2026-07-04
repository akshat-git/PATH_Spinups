#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build h-optimus.sif h-optimus.def

# Run the spin script inside the container:
#   apptainer exec h-optimus.sif python h-optimus_spin.py
