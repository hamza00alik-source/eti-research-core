"""Export trusted ETI candidates to a verified ONNX mobile inference bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .model import BaselineModel


def export_model_to_onnx(
    model: BaselineModel,
    feature_names: list[str],
    output_dir: str | Path,
    *,
    artifact_sha256: str | None = None,
    target_opset: int = 17,
) -> dict[str, str | dict]:
    """Convert a fitted ETI model and verify ONNX Runtime parity before packaging.

    The ONNX graph emits raw classifier probabilities. ETI's chronological
    temperature-calibration value is stored in metadata and applied by the
    Android inference wrapper after ONNX execution.
    """
    if not isinstance(model, BaselineModel) or model.classes_ is None:
        raise ValueError("a fitted ETI BaselineModel is required")
    if not isinstance(feature_names, list) or not feature_names or any(
        not isinstance(name, str) or not name for name in feature_names
    ):
        raise ValueError("feature_names must be a non-empty list of names")
    estimator = model.estimator
    if int(getattr(estimator, "n_features_in_", -1)) != len(feature_names):
        raise ValueError("feature schema length does not match fitted model input width")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be unique")
    try:
        import onnx
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional mobile exporter dependencies: pip install -r requirements-mobile.txt"
        ) from exc

    # skl2onnx's HistGradientBoosting converter can emit Python bool values
    # inside ONNX INT attributes (e.g. missing-value routing flags). ONNX's
    # protobuf correctly rejects bool for repeated int fields. Normalize only
    # those boolean attribute values at the converter boundary.
    import skl2onnx.common._container as converter_container

    original_make_node = converter_container.make_node

    def _make_node_with_integer_boolean_attrs(op_type, inputs, outputs, **attrs):
        normalized = {}
        for key, value in attrs.items():
            if isinstance(value, (list, tuple)) and any(type(item) is bool for item in value):
                value = [int(item) if type(item) is bool else item for item in value]
            elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
                value = value.astype(np.int64)
            normalized[key] = value
        return original_make_node(op_type, inputs, outputs, **normalized)

    converter_container.make_node = _make_node_with_integer_boolean_attrs
    try:
        converted = convert_sklearn(
            estimator,
            initial_types=[("features", FloatTensorType([None, len(feature_names)]))],
            options={id(estimator): {"zipmap": False}},
            target_opset=target_opset,
        )
    finally:
        converter_container.make_node = original_make_node
    onnx.checker.check_model(converted)
    model_bytes = converted.SerializeToString()

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    if len(output_names) < 2:
        raise RuntimeError("converted classifier does not expose a probability output")
    probability_output_index = 1

    probe = np.random.default_rng(20261114).normal(
        loc=0.0, scale=1.0, size=(32, len(feature_names))
    ).astype(np.float32)
    probe_frame = pd.DataFrame(probe, columns=feature_names)
    expected = np.asarray(estimator.predict_proba(probe_frame), dtype=np.float32)
    actual_outputs = session.run(None, {input_name: probe})
    actual = np.asarray(actual_outputs[probability_output_index], dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=1e-4, atol=1e-4
    ):
        raise RuntimeError("ONNX probabilities failed parity verification against ETI's source model")
    if not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
        raise RuntimeError("ONNX runtime returned invalid probability values")

    temperature = model.calibration_temperature()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "eti_model.onnx"
    metadata_path = output / "eti_model_metadata.json"
    onnx_path.write_bytes(model_bytes)
    onnx_sha256 = hashlib.sha256(model_bytes).hexdigest()
    metadata = {
        "format": "onnx",
        "model_family": "HistGradientBoostingClassifier",
        "input_name": input_name,
        "input_dtype": "float32",
        "input_shape": [-1, len(feature_names)],
        "feature_names": feature_names,
        "class_ids": [int(value) for value in estimator.classes_],
        "class_meanings": {"0": "short", "1": "flat", "2": "long"},
        "probability_output_index": probability_output_index,
        "probability_output_name": output_names[probability_output_index],
        "calibration": {
            "status": model.calibration_status,
            "temperature": temperature,
            "method": "temperature_scaling" if temperature is not None else None,
            "applied_by": "mobile_runtime_after_onnx",
        },
        "source_artifact_sha256": artifact_sha256,
        "onnx_sha256": onnx_sha256,
        "target_opset": target_opset,
        "verification": {
            "runtime": "onnxruntime-cpu",
            "probe_rows": int(len(probe)),
            "probability_parity": "passed",
            "absolute_tolerance": 1e-4,
            "relative_tolerance": 1e-4,
        },
        "safety": {
            "candidate_only": True,
            "model_approval": "not_granted",
            "live_trading_enabled": False,
            "warning": "ONNX export is an inference format, not evidence of model quality or profitability.",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bundle_path = output / "eti_mobile_model_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(onnx_path, arcname=onnx_path.name)
        archive.write(metadata_path, arcname=metadata_path.name)
    return {
        "onnx_path": str(onnx_path),
        "metadata_path": str(metadata_path),
        "bundle_path": str(bundle_path),
        "onnx_sha256": onnx_sha256,
        "metadata": metadata,
    }
