# HR

MLB home-run probability model workspace.

This repository is being initialized from the `hr_model_v1.1` package for audit, repair, reproducible training, and eventually pregame scoring.

## Current status

The imported v1.1 package contains a real Statcast-based modeling pipeline, but it is **not yet accepted as statistically clean**. A September 2026 audit identified critical historical feature-construction issues, including same-season temporal leakage in pitcher arsenal/usage and park-factor features, plus plate-appearance and pitch-type attribution concerns.

The 2025 holdout has **not been scored or evaluated** and should remain sealed while the 2015–2024 historical pipeline is repaired.

Development should happen on feature branches and be promoted to `main` only after review.
