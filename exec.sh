#!/bin/bash

OUTPUT_FILE="output.log"

echo "=== Feature Extraction ===" | tee "$OUTPUT_FILE"
python3 feature_extraction.py 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"

echo "=== KNN Classification ===" | tee -a "$OUTPUT_FILE"
python3 knn.py 2>&1 | tee -a "$OUTPUT_FILE"
