#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build virchow.sif virchow.def

# Run the spin script inside the container:
#   apptainer exec virchow.sif python virchow_spin.py
