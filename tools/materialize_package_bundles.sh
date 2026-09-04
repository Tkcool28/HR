#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
BUNDLES="$ROOT/package_bundles"
WORK="$ROOT/.package_restore"

required_bundles=(
  00_code_tests_reports.tar.gz
  01_curated_small_and_splits.tar.gz
  02_bip_all.tar.gz
  03_raw_pa_2015_2017.tar.gz
  04_raw_pa_2018_2020.tar.gz
  05_raw_pa_2021_2024.tar.gz
  10_curated_pa_00.tar.gz
  10_curated_pa_01.tar.gz
  10_curated_pa_02.tar.gz
  10_curated_pa_03.tar.gz
  20_features_v1_00.tar.gz
  20_features_v1_01.tar.gz
  20_features_v1_02.tar.gz
  30_features_v1_1_00.tar.gz
  30_features_v1_1_01.tar.gz
  30_features_v1_1_02.tar.gz
  30_features_v1_1_03.tar.gz
  30_features_v1_1_04.tar.gz
  99_manifest_and_reconstruction.tar.gz
)

for f in "${required_bundles[@]}"; do
  [[ -f "$BUNDLES/$f" ]] || { echo "MISSING_BUNDLE: $f"; exit 2; }
done

rm -rf "$WORK"
mkdir -p "$WORK/base" "$WORK/pa" "$WORK/v1" "$WORK/v11"

# Extract logical bundles directly into the repository workspace.
for f in \
  00_code_tests_reports.tar.gz \
  01_curated_small_and_splits.tar.gz \
  02_bip_all.tar.gz \
  03_raw_pa_2015_2017.tar.gz \
  04_raw_pa_2018_2020.tar.gz \
  05_raw_pa_2021_2024.tar.gz; do
  tar -xzf "$BUNDLES/$f" -C "$WORK/base"
done

# The logical bundles contain an hr_model/ root. Overlay its contents onto the checkout.
if [[ -d "$WORK/base/hr_model" ]]; then
  cp -a "$WORK/base/hr_model/." "$ROOT/"
else
  echo "Expected hr_model/ root not found after extracting logical bundles" >&2
  exit 3
fi

# Extract multipart raw-byte pieces for the three large parquet files.
for f in "$BUNDLES"/10_curated_pa_*.tar.gz; do tar -xzf "$f" -C "$WORK/pa"; done
for f in "$BUNDLES"/20_features_v1_*.tar.gz; do tar -xzf "$f" -C "$WORK/v1"; done
for f in "$BUNDLES"/30_features_v1_1_*.tar.gz; do tar -xzf "$f" -C "$WORK/v11"; done

mkdir -p "$ROOT/data/curated" "$ROOT/features/v1" "$ROOT/features/v1.1"
cat "$WORK"/pa/part_*  > "$ROOT/data/curated/pa.parquet"
cat "$WORK"/v1/part_*  > "$ROOT/features/v1/game_features.parquet"
cat "$WORK"/v11/part_* > "$ROOT/features/v1.1/game_features.parquet"

expected_pa="39de683b558294111ad847aef50aa627a75c1f5062a8c92a34caefd28f6099ae"
expected_v1="e553a4c068f52ea4b6b775628d649fe945b551aa3469ea19df0a319d0334202f"
expected_v11="a04531d916bdd96f7753bb0630f11d42edec4a275591d3d204cb04287355b0a6"

check_hash() {
  local expected="$1" path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH_MISMATCH: $path expected=$expected actual=$actual" >&2
    exit 4
  fi
  echo "HASH_OK: $path $actual"
}

check_hash "$expected_pa"  "$ROOT/data/curated/pa.parquet"
check_hash "$expected_v1"  "$ROOT/features/v1/game_features.parquet"
check_hash "$expected_v11" "$ROOT/features/v1.1/game_features.parquet"

echo "PACKAGE_MATERIALIZATION_COMPLETE"
