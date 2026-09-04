# v1.1 package import status

Source archive inspected locally: `hr_model_v1.1.tar.gz`

Archive SHA-256:

`f738b66263d2efc6bb6ddd85c171ac1a69457e9b8536c58d01f81964038f50d9`

The full per-file SHA-256 manifest is in `docs/original_package_sha256.txt`.

## Imported on this branch

- repository audit README / status
- `.gitignore` guardrails for heavyweight generated artifacts
- explicit Python dependencies including `pyarrow`
- GitHub Actions audit workflow
- `tools/inspect_parquet.py`
- original v1.1 package README preserved under `docs/`
- original package hash manifest
- original data acquisition / processing source currently imported:
  - `src/data/acquire.py`
  - `src/data/acquire_bip.py`
  - `src/data/process.py`
  - `src/data/process_season.py`
  - `src/data/finalize.py`
  - `src/data/process_bip.py`

The source is intentionally imported before methodology fixes so the delivered v1.1 baseline remains auditable.

## Original archive size

Approximately 252 MB extracted / 217 MB compressed.

Largest files include:

- `features/v1.1/game_features.parquet` — 76,981,178 bytes
- `data/curated/pa.parquet` — 59,430,122 bytes
- `features/v1/game_features.parquet` — 44,070,844 bytes
- `data/raw/bip_all.parquet` — 18,669,602 bytes
- annual raw PA parquets — roughly 2.6–6.9 MB each

## Heavy data policy

These heavyweight generated Parquet/model artifacts are deliberately not being added to ordinary Git history during the initial import.

The audit workflow therefore distinguishes:

1. **code smoke** — dependency installation, Python compilation, import checks;
2. **data-backed audit** — Parquet inventory plus the original integrity tests, but only when the complete required data surface is present on the runner.

The workflow must not claim that the model/data passed full testing when the Parquet artifacts are absent.

## 2025 holdout

2025 has not been scored/evaluated for the current repair work and remains outside the present methodology-repair scope. Historical repair work should focus on 2015–2024 first.

Knowing the announced pregame starting pitcher or lineup is not itself look-ahead. Historical reconstruction should, however, represent the pregame starter/lineup rather than derive matchup identity from how the completed game unfolded.

## Remaining baseline import

Still to import verbatim before repair work begins:

- feature-building source and metadata
- training source
- original integrity tests
- original text reports / logs
- small non-generated metadata/CSV artifacts where useful

Then establish a safe materialization strategy for the large Parquet data so Actions can inspect schemas/rows and run the full original tests.
