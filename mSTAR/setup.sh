#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build mSTAR.sif mSTAR.def

# Run the spin script inside the container:
#   apptainer exec mSTAR.sif python mSTAR_spin.py
