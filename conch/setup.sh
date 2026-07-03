#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build conch.sif conch.def

# Run the spin script inside the container:
#   apptainer exec conch.sif python conch_spin.py
