# Gemini CLI Context: Phishing URL Classifier (ID3 + Streamlit)

This project is a Phishing URL Classifier built with a custom ID3 decision tree implementation and a Streamlit-based user interface. It is designed to illustrate an end-to-end data science pipeline for educational purposes, focusing on the PhiUSIIL Phishing URL Dataset.

## Project Overview

- **Purpose:** Classify URLs as **phishing** or **legitimate** using machine learning.
- **Label Mapping (important):**
    - `1` = **legitimate**
    - `0` = **phishing**
- **Main Technologies:** 
    - **Python:** Core programming language.
    - **Streamlit:** Interactive web application for the UI.
    - **ID3 Algorithm:** A custom implementation of the ID3 decision tree (using entropy and information gain), specifically designed for discrete/categorical features.
    - **Pandas & NumPy:** Data manipulation and numerical computation.
    - **scikit-learn:** Used for data splitting, evaluation metrics, and discretization support.
    - **Plotly:** Interactive charts for data visualization and model evaluation.
    - **Graphviz:** Decision tree rendering in UI sections that visualize tree structure.
    - **Pytest:** Automated testing framework.
- **Architecture:** 
    - `app.py`: Entry point for the Streamlit application.
    - `src/`: Core logic and services.
        - `id3.py`: Implementation of the ID3 classifier.
        - `preprocessing.py`: Data cleaning, categorical grouping, and numeric discretization pipeline.
        - `services/`: Business logic orchestration (data loading, training).
        - `ui/`: Modular UI components and dashboard sections.
    - `tests/`: Comprehensive test suite for algorithm correctness and pipeline stability.

## Building and Running

### Prerequisites
- Python 3.9+ (suggested)
- A virtual environment is recommended.

### Installation
```powershell
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install -r requirements.txt
```

### Running the Application
```powershell
streamlit run app.py
```
The application will be accessible via your browser (usually at `http://localhost:8501`). You can upload a PhiUSIIL CSV or use the default dataset if present in `data/`.

### Running Tests
```powershell
pytest
```

## Development Conventions

- **Custom ID3 Implementation:** Do NOT replace `src/id3.py` with standard library classifiers (like scikit-learn's `DecisionTreeClassifier`) unless specifically requested, as the project's educational focus is on the manual ID3 implementation.
- **Data Preprocessing:** All numeric data MUST be discretized before being passed to the ID3 model. Use the `PreprocessingPipeline` in `src/preprocessing.py` for consistent transformations across training and prediction.
- **Multilingual Support:** UI strings are managed through a translation helper `L(lang, en, vi)` in `src/ui/common.py`. Adhere to this pattern when adding new UI elements.
- **Language Default:** UI currently defaults to **Tiếng Việt** in the language selector.
- **Label Semantics Safety:** Keep class meaning consistent everywhere (`1=legitimate`, `0=phishing`) when editing metrics, reports, and user-facing labels.
- **Configuration:** Use `src/config.py` and environment variables (e.g., `PHISHING_DEBUG_ERRORS`) for runtime toggles.
- **Testing:** New features or bug fixes should include corresponding tests in the `tests/` directory.

## Key Files
- `app.py`: Main Streamlit app logic.
- `src/id3.py`: Core ID3 algorithm logic (Entropy, Information Gain, Tree Building).
- `src/preprocessing.py`: Handles feature selection, missing values, and discretization.
- `src/services/training_service.py`: Orchestrates the entire training flow.
- `src/data_loader.py`: Dataset loading and schema validation.
- `src/utils.py`: Constants and utility functions (e.g., label mapping).
