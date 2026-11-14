import numpy as np
import pandas as pd
import pytest

from eti.model import BaselineModel
from eti.mobile_export import export_model_to_onnx


def test_mobile_export_has_onnx_probability_parity(tmp_path):
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    rng = np.random.default_rng(17)
    x = rng.normal(size=(600, 4))
    score = x[:, 0] + 0.3 * x[:, 1]
    y = np.where(score < -0.35, 0, np.where(score > 0.35, 2, 1))
    columns = ["ret_1", "ret_3", "volatility_20", "rsi_14"]
    model = BaselineModel().fit(pd.DataFrame(x, columns=columns), y)
    result = export_model_to_onnx(model, columns, tmp_path, artifact_sha256="a" * 64)

    assert (tmp_path / "eti_model.onnx").is_file()
    assert (tmp_path / "eti_model_metadata.json").is_file()
    assert (tmp_path / "eti_mobile_model_bundle.zip").is_file()
    assert result["metadata"]["verification"]["probability_parity"] == "passed"
    assert result["metadata"]["calibration"]["status"] == "uncalibrated"
    assert result["metadata"]["safety"]["live_trading_enabled"] is False


def test_mobile_export_rejects_mismatched_feature_schema(tmp_path):
    model = BaselineModel()
    model.fit(pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], columns=["a", "b"]), [0, 1, 2])
    with pytest.raises(ValueError, match="schema length"):
        export_model_to_onnx(model, ["a"], tmp_path)
