#!/bin/bash

export EXECUTION_DATE=$(date +"%Y-%m-%d")
export MODE="dev"

poetry shell

echo "Running for date: $EXECUTION_DATE"
PYTHONPATH=. python main.py load-and-preprocess-data --train-size 0.7 --val-size 0.10 --test-size 0.20
PYTHONPATH=. python main.py train-and-evaluate-model --model-name catboost --n-trials 1 --scoring accuracy 