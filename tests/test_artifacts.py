import numpy as np
import pandas as pd
import pytest

from eti.artifacts import ModelArtifactError, ModelArtifactStore
from eti.model import BaselineModel


def _fitted_model():
    x = pd.DataFrame({
        "feature_a": np.linspace(-1.0, 1.0, 90),
        "feature_b": np.cos(np.linspace(0.0, 6.0, 90)),
    })
    y = np.tile(np.array([0, 1, 2]), 30)
    return BaselineModel().fit(x, y), x


def test_content_addressed_artifact_round_trip_and_prediction_parity(tmp_path):
    model, frame = _fitted_model()
    store = ModelArtifactStore(tmp_path / "artifacts")
    reference = store.save_model(model)
    assert reference["artifact_uri"].startswith("artifact://sha256/")
    assert len(reference["artifact_sha256"]) == 64

    restored = store.load_model(reference["artifact_uri"], reference["artifact_sha256"])
    np.testing.assert_allclose(
        restored.predict_proba(frame),
        model.predict_proba(frame),
        rtol=0,
        atol=0,
    )


def test_artifact_store_rejects_unfitted_models_and_unsafe_references(tmp_path):
    store = ModelArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="fitted"):
        store.save_model(BaselineModel())
    with pytest.raises(ModelArtifactError, match="unsupported"):
        store.load_model("artifact://../../outside", "a" * 64)


def test_artifact_checksum_is_verified_before_deserialization(tmp_path):
    model, _ = _fitted_model()
    store = ModelArtifactStore(tmp_path / "artifacts")
    reference = store.save_model(model)
    path = store._path_for_digest(reference["artifact_sha256"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ModelArtifactError, match="checksum"):
        store.load_model(reference["artifact_uri"], reference["artifact_sha256"])


def test_artifact_store_rejects_symlinked_digest_directory(tmp_path):
    model, _ = _fitted_model()
    store = ModelArtifactStore(tmp_path / "artifacts")
    reference = store.save_model(model)
    digest = reference["artifact_sha256"]
    real_directory = store.root / digest[:2]
    moved_directory = store.root / "real"
    real_directory.rename(moved_directory)
    real_directory.symlink_to(moved_directory, target_is_directory=True)

    with pytest.raises(ModelArtifactError, match="symbolic-link"):
        store.load_model(reference["artifact_uri"], digest)
