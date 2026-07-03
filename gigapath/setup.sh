#!/usr/bin/env bash
set -e

# Build the container (run once)
apptainer build gigapath.sif gigapath.def

# Run the spin script inside the container:
#   apptainer exec gigapath.sif python gigapath_spin.py
