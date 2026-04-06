"""Schema validation for CSV load."""

import pandas as pd
import pytest

from src.data_loader import schema_required_columns, validate_schema
from src.utils import FEATURE_COLUMNS, TARGET_COL


def test_validate_schema_missing_target():
    df = pd.DataFrame({c: [0] for c in FEATURE_COLUMNS[:3]})
    with pytest.raises(ValueError, match="target"):
        validate_schema(df)


def test_validate_schema_missing_feature():
    cols = [TARGET_COL, *list(FEATURE_COLUMNS)[:-1]]
    df = pd.DataFrame({c: [0] for c in cols})
    with pytest.raises(ValueError, match="feature columns"):
        validate_schema(df)


def test_validate_schema_filename_optional():
    df = pd.DataFrame({TARGET_COL: [0, 1], **{c: [0, 0] for c in FEATURE_COLUMNS}})
    validate_schema(df)


def test_schema_required_columns_includes_all_features():
    req = schema_required_columns()
    assert req[0] == TARGET_COL
    assert set(req) == {TARGET_COL, *FEATURE_COLUMNS}
