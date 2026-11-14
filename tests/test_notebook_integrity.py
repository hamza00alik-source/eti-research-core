import json
from pathlib import Path


def test_canonical_kaggle_notebook_is_valid_and_uses_nested_pipeline():
    notebook = json.loads(Path("notebooks/ETI_v0_1_Training.ipynb").read_text())
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "--evaluation" in code
    assert "nested" in code
    assert "--symbol" in code
    assert "MAX_MODEL_FITS = 32" in code
    assert "--max-model-fits" in code
