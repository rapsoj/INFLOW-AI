# INFLOW-AI

INFLOW-AI is a flood inundation prediction workflow for the INFLOW study area. It refreshes temporal inputs, runs the prediction pipeline, exports forecast CSVs and plots, generates spatial flood outputs, and produces explanation charts.

## Quickstart

1. Create and activate a Python 3.11+ virtual environment.
2. Install the project from the repository root:

```bash
pip install -e .
```

3. Make sure the required local data folders are available, especially `data/`, `model/`, and `predictions/`.
4. Run the pipeline:

```bash
python __main__.py
```

   You can also use the wrapper script:

```bash
python scripts/run_pipeline.py
```

## Configuration

Default paths and runtime constants live in [config/defaults.toml](config/defaults.toml). They can be overridden with environment variables using the `INFLOW_AI_*` prefix.

Examples:

```bash
export INFLOW_AI_DATA_DIR=/path/to/data
export INFLOW_AI_PREDICTIONS_DIR=/path/to/predictions
export INFLOW_AI_TEMPORAL_MODEL=/path/to/temporal_model.keras
```

## Repository Layout

```text
├── __main__.py
├── config/
│   └── defaults.toml
├── explanations/
├── model/
├── predictions/
├── processing/
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── inflow_ai/
├── pyproject.toml
├── requirements.txt
└── README.md
```

The code is organized into a `src`-style package:

- `src/inflow_ai/pipelines/` for orchestration
- `src/inflow_ai/data/` for ingestion and cleaning utilities
- `src/inflow_ai/models/` for model training and inference
- `src/inflow_ai/explainability/` for explanation and plotting helpers

The legacy top-level modules remain in place for compatibility, but `src/inflow_ai` is the preferred import surface.

## Inputs and Outputs

- Temporal source data is read from `data/historic/` and `data/stats/` by default.
- Model artifacts are read from `model/`.
- Forecast runs are written under `predictions/inundation_predictions_<start>_to_<end>/`.
- Spatial prediction artifacts are written under each run’s `spatial_predictions/` folder.
- Explanation charts are written under each run’s `explanations/` folder.

## Packaging

Project metadata and pinned runtime dependencies are defined in [pyproject.toml](pyproject.toml). `requirements.txt` is still present for compatibility, but `pyproject.toml` is now the canonical project definition.

## Notes

- The repository expects the data and model assets referenced above to exist locally before running the pipeline.
- If you change defaults, prefer editing [config/defaults.toml](config/defaults.toml) or setting `INFLOW_AI_*` environment variables rather than editing code.