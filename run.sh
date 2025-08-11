#!/bin/bash
export PYTHONPATH=.
ulimit -n 32000 # Increase ability to leave open files, for running many workers (e.g. 32+).
python nmaipy/exporter.py \
    --aoi-file "s3://data-extraction-ap-southeast-2/DataExtraction/TRO-3810_QT_Melbourne_7/TRO-3810_QT_Melbourne_7_parcels.parquet" \
    --output-dir "data/outputs" \
    --country au \
    --processes 4 \
    --system-version-prefix "gen6-" \
    --packs vegetation surfaces \
    --include-parcel-geometry \
    --save-features \
    --aoi-grid-inexact \
    --aoi-grid-min-pct 0 \
    --no-parcel-mode \
    --chunk-size 100
