#!/usr/bin/env bash

CHROMALXE=/home/youngsam/sw/chroma-lxe

python $CHROMALXE/macros/nphoton_scan.py \
            --config $CHROMALXE/geometry/config/detector.yaml \
            --positions $CHROMALXE/data/lightmap_points_2mm_orthofill.npy \
            --output $CHROMALXE/data/results/nphoton_scan_2mm_orthofill.h5 \
            -N 1_000_000

python $CHROMALXE/macros/nphoton_scan.py \
            --config $CHROMALXE/geometry/config/detector.yaml \
            --positions $CHROMALXE/data/lightmap_points_2.5mm_orthofill.npy \
            --output $CHROMALXE/data/results/nphoton_scan_2.5mm_orthofill.h5 \
            -N 1_000_000