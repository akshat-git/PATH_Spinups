#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build virchow2.sif virchow2.def

# Run the spin script inside the container:
#   apptainer exec virchow2.sif python virchow2_spin.py
