from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import PreprocessConfig, PreprocessingPipeline
from src.utils import TARGET_COL


def test_transform_before_fit_raises():
    pipe = PreprocessingPipeline(config=PreprocessConfig())
    with pytest.raises(RuntimeError, match="Call fit"):
        pipe.transform_X(pd.DataFrame({"TLD": ["com"], "URLLength": [1.0]}))


def test_transform_missing_feature_columns_raises():
    train = pd.DataFrame({TARGET_COL: [0, 1], "TLD": ["com", "org"], "URLLength": [1.0, 2.0]})
    pipe = PreprocessingPipeline(config=PreprocessConfig())
    pipe.fit(train)
    with pytest.raises(ValueError, match="missing columns"):
        pipe.transform_X(pd.DataFrame({"TLD": ["com"]}))


def test_manual_input_frame_before_fit_raises():
    pipe = PreprocessingPipeline(config=PreprocessConfig())
    with pytest.raises(RuntimeError, match="Call fit"):
        pipe.manual_input_frame({"URLLength": 5.0})

