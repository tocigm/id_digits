#!/bin/bash
# Add dependency onto projects
PROJ_DIR=$HOME/WORK/projects/bagiks/digits/code
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH

python3 $PROJ_DIR/run.py
