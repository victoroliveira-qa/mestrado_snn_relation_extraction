import pandas as pd
import pytest
from src.embeddings.model_registry import build_texts

def test_text_builder():
    df = pd.DataFrame({"sentence_text": ["hello"], "mention_text": ["world"]})
    result = build_texts(df, {"text_columns": ["sentence_text", "mention_text"]})
    assert result == ["sentence_text: hello\nmention_text: world"]

def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing source columns"):
        build_texts(pd.DataFrame({"x": ["a"]}), {"text_columns": ["missing"]})
