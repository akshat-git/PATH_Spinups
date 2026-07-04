#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build hipt.sif hipt.def

# Run the spin script inside the container:
#   apptainer exec hipt.sif python hipt_spin.py
