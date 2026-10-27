#!/usr/bin/env sh
set -eu
config="${1:-configs/default.json}"
for stage in label samples train analyze; do
  python -m traffic_prediction "$stage" --config "$config"
done
