#!/bin/bash
# Script to run the MetaDSL demo properly

cd "$(dirname "$0")"
echo "Running MetaDSL Demo with mix..."
mix run self_modifying_metadsl_example.exs "$@"