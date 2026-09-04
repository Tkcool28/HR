# Package bundles

Place the 19 repackaged `hr_model_v1.1` tarballs in this directory **without renaming them**.

Expected filenames:

- `00_code_tests_reports.tar.gz`
- `01_curated_small_and_splits.tar.gz`
- `02_bip_all.tar.gz`
- `03_raw_pa_2015_2017.tar.gz`
- `04_raw_pa_2018_2020.tar.gz`
- `05_raw_pa_2021_2024.tar.gz`
- `10_curated_pa_00.tar.gz`
- `10_curated_pa_01.tar.gz`
- `10_curated_pa_02.tar.gz`
- `10_curated_pa_03.tar.gz`
- `20_features_v1_00.tar.gz`
- `20_features_v1_01.tar.gz`
- `20_features_v1_02.tar.gz`
- `30_features_v1_1_00.tar.gz`
- `30_features_v1_1_01.tar.gz`
- `30_features_v1_1_02.tar.gz`
- `30_features_v1_1_03.tar.gz`
- `30_features_v1_1_04.tar.gz`
- `99_manifest_and_reconstruction.tar.gz`

Once all bundles are present, `.github/workflows/audit.yml` runs `tools/materialize_package_bundles.sh` automatically. The script:

1. extracts the logical bundles;
2. reconstructs the three large split Parquet files;
3. verifies their exact original SHA-256 hashes;
4. exposes the restored package to the Parquet inventory and original integrity tests.

Do not extract or modify the tarballs before upload.
