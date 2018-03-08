#!/usr/bin/env bash
VERBOSE=-1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -v -s . -p "*_tests.py"
