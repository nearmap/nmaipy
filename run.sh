#!/bin/bash
export PYTHONPATH=.
ulimit -n 32000 # Increase ability to leave open files, for running many workers (e.g. 32+).
python /home/jovyan/Development/datascience/nmaipy/scripts/ai_offline_parcel.py \
    --parcel-dir "data/parcels" \
    --output-dir "data/outputs" \
    --country us \
    --workers 4 \
    --system-version-prefix "gen6-" \
    --packs building vegetation \
    --include-parcel-geometry \
    --save-features