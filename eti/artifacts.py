"""Immutable content-addressed storage for trusted ETI model artifacts.

Only server-generated BaselineModel objects may be saved. Artifacts are never
accepted as user uploads. Loading joblib/pickle can execute code, so the
artifact directory must remain writable only by the trusted training service.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path

import joblib

from .model import BaselineModel


_ARTIFACT_URI = re.compile(r"^artifact://sha256/([a-f0-9]{64})$")


class ModelArtifactError(RuntimeError):
    """Artifact is absent, malformed or fails its integrity check."""


class ModelArtifactStore:
    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root or os.environ.get("ETI_ARTIFACT_DIR", "data/artifacts")).expanduser().resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _path_for_digest(self, digest: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ModelArtifactError("invalid artifact digest")
        directory = self.root / digest[:2]
        path = directory / (digest + ".joblib")
        if path.is_symlink() or directory.is_symlink():
            raise ModelArtifactError("symbolic-link artifacts are not accepted")
        return path

    def save_model(self, model: BaselineModel) -> dict[str, str]:
        if not isinstance(model, BaselineModel) or model.classes_ is None:
            raise ValueError("only a fitted ETI BaselineModel can be stored")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        fd, temporary_name = tempfile.mkstemp(prefix=".eti-model-", suffix=".joblib", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            joblib.dump(model, temporary, compress=3)
            digest = self._sha256(temporary)
            destination = self._path_for_digest(digest)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(destination.parent, 0o700)
            except OSError:
                pass
            if destination.exists():
                if destination.is_symlink() or not hmac.compare_digest(self._sha256(destination), digest):
                    raise ModelArtifactError("existing content-addressed artifact failed integrity check")
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, destination)
                try:
                    os.chmod(destination, 0o600)
                except OSError:
                    pass
            return {
                "artifact_uri": "artifact://sha256/" + digest,
                "artifact_sha256": digest,
            }
        finally:
            temporary.unlink(missing_ok=True)

    def load_model(self, artifact_uri: str, expected_sha256: str) -> BaselineModel:
        match = _ARTIFACT_URI.fullmatch(artifact_uri) if isinstance(artifact_uri, str) else None
        if not match or not isinstance(expected_sha256, str):
            raise ModelArtifactError("unsupported or malformed artifact reference")
        digest = match.group(1)
        if not hmac.compare_digest(digest, expected_sha256):
            raise ModelArtifactError("artifact reference and registry checksum do not match")
        path = self._path_for_digest(digest)
        if not path.is_file() or path.is_symlink():
            raise ModelArtifactError("model artifact is not available on this server")
        actual = self._sha256(path)
        if not hmac.compare_digest(actual, digest):
            raise ModelArtifactError("model artifact checksum verification failed")
        try:
            # Safe only because this directory is server-owned and never accepts uploads.
            model = joblib.load(path)
        except Exception as exc:
            raise ModelArtifactError("model artifact could not be deserialized") from exc
        if not isinstance(model, BaselineModel) or model.classes_ is None:
            raise ModelArtifactError("artifact is not a fitted ETI BaselineModel")
        return model
