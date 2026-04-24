# AGENTS.md

Guidance for AI agents and contributors working in this repo.

## Project summary

This is an academic Streamlit app that classifies URLs as **phishing** vs **legitimate** using a **custom ID3 decision tree** (entropy + information gain). Avoid replacing the custom ID3 implementation with scikit-learn’s `DecisionTreeClassifier`—the point is to keep the tree explainable and “from scratch”.

### Label convention (critical)

- Dataset/Project mapping is:
  - `label = 1` -> **legitimate**
  - `label = 0` -> **phishing**
- Keep this convention consistent across:
  - preprocessing/normalization
  - evaluation metrics (positive class choices)
  - UI texts and prediction explanations
  - tests

## Quickstart (local dev)

### Setup

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv venv
```

Windows (PowerShell):

```bash
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install deps:

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

## Tests

```bash
python -m pytest
```

## Configuration (environment variables)

- **`PHISHING_DEBUG_ERRORS`**: default `false`. Truthy values are: `1`, `true`, `yes`, `on` (case-insensitive). When enabled, the UI shows detailed exception output for debugging.
- **`PHISHING_DEFAULT_ROW_LIMIT`**: default `8000`. Integer row-limit used in training controls; `0` means “use all rows”.

## Repo map (important paths)

- **`app.py`**: Streamlit entry point (UI wiring + sections).
- **`src/config.py`**: typed runtime settings read from env vars.
- **`src/services/`**: “orchestration” layer
  - `src/services/data_service.py`: dataset loading (upload vs default file).
  - `src/services/training_service.py`: preprocessing + training runner and artifacts.
- **`src/preprocessing.py`**: preprocessing pipeline (impute, categorical handling, discretization).
- **`src/id3.py`**: custom ID3 decision tree implementation.
- **`src/evaluation.py`**: metrics and evaluation helpers.
- **`src/predictor.py`**: prediction + explanation helpers (decision path / rules).
- **`src/ui/`** and **`src/ui/sections/`**: Streamlit UI components/sections.
- **`data/`**: optional default dataset location (the app looks for `data/PhiUSIIL_Phishing_URL_Dataset.csv`).

### Visualization note

- Decision tree visualization is currently standardized on **Graphviz** in UI sections that render the tree.

## Agent/contributor guidelines

- **Keep preprocessing consistent**: fit all preprocessing artifacts on train data, then apply to test/predict (avoid leakage).
- **Don’t break the dataset contract**: the target column is `label`; `FILENAME` is metadata (not a feature).
- **Respect label semantics**: do not accidentally swap class meaning (`1=legitimate`, `0=phishing`) in docs/UI/tests.
- **Prefer small changes**: make changes in one layer at a time (UI vs services vs core algorithm) and keep diffs reviewable.
- **Dependencies**: if you add a new dependency, update `requirements.txt` and justify why it’s needed.
- **UI sanity check**: after UI/service changes, ensure the app still starts with `streamlit run app.py` and the sidebar upload/default-data flow still works.
