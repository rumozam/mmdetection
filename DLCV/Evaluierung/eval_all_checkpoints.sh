#!/bin/bash

CONFIG=$1
CHECKPOINT_DIR=$2
WORK_DIR=$3

EPOCHS="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300"

for epoch in $EPOCHS
do
    checkpoint=$CHECKPOINT_DIR/epoch_$epoch.pth
    echo $checkpoint
    python tools/test.py $CONFIG $checkpoint --eval bbox --work-dir $WORK_DIR
done