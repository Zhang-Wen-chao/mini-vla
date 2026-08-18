#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
if [ -d data/pusht_cchi_v7_replay.zarr ]; then
  echo "already downloaded: data/pusht_cchi_v7_replay.zarr"
  exit 0
fi

echo "downloading pusht.zip (~1.5 GB) ..."
curl -L -o data/pusht.zip https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
echo "extracting ..."
unzip -q data/pusht.zip -d data
rm -f data/pusht.zip
echo "done: data/pusht_cchi_v7_replay.zarr"
