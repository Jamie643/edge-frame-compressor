from pathlib import Path

import numpy as np
import pytest

from edge_frame_compressor import (
    BoundingBox,
    DetectionInput,
    EdgeFrameCompressor,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yunet.onnx"
MODEL_HASH = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class _FakeYuNet:
    """Stands in for cv2.FaceDetectorYN so tests don't need the ONNX model."""

    def __init__(self, faces):
        self._faces = faces

    def setInputSize(self, size):  # noqa: N802 (mimics OpenCV API)
        pass

    def detect(self, frame):
        if self._faces is None:
            return 1, None
        return 1, np.array(self._faces, dtype=np.float32)


def _face_row(x, y, w, h):
    # YuNet row layout: x, y, w, h, 5 landmark (x,y) pairs, score
    return [x, y, w, h] + [0.0] * 10 + [0.99]


def _compressor_with_faces(faces):
    c = EdgeFrameCompressor(require_redaction=False)
    c.yunet_detector = _FakeYuNet(faces)
    c.redaction_enabled = True
    return c


def test_solid_fill_redaction_uses_injected_detector():
    c = _compressor_with_faces([_face_row(50, 50, 50, 50)])

    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    redacted, count = c.redact_pii(frame)

    assert count == 1, "expected exactly one redaction"
    assert np.all(redacted[50:100, 50:100] == 0), "face region must be black"
    assert np.all(redacted[0:50, 0:50] == 255), "outside region must be untouched"
    assert np.all(frame[50:100, 50:100] == 255), "input frame must not be mutated"


def test_no_faces_no_redaction():
    c = _compressor_with_faces(None)
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    redacted, count = c.redact_pii(frame)
    assert count == 0
    assert np.array_equal(redacted, frame)


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (BoundingBox(x=50, y=50, w=50, h=50), (50, 50, 50, 50)),
        (BoundingBox(x=-10, y=50, w=50, h=50), (0, 50, 40, 50)),
        (BoundingBox(x=180, y=50, w=50, h=50), (180, 50, 20, 50)),
        (BoundingBox(x=500, y=50, w=50, h=50), None),
        (BoundingBox(x=50, y=50, w=0, h=50), None),
        (BoundingBox(x=50, y=50, w=-5, h=50), None),
    ],
)
def test_clamp_bbox(bbox, expected):
    c = EdgeFrameCompressor(require_redaction=False)
    assert c._clamp_bbox(bbox, img_w=200, img_h=200) == expected


def test_require_redaction_default_true_raises_without_model():
    with pytest.raises(ValueError):
        EdgeFrameCompressor()


def test_missing_model_file_raises(tmp_path):
    fake = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        EdgeFrameCompressor(
            yunet_model_path=str(fake),
            yunet_model_sha256="0" * 64,
        )


def test_bad_hash_raises(tmp_path):
    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"not a real model")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        EdgeFrameCompressor(
            yunet_model_path=str(fake),
            yunet_model_sha256="0" * 64,
        )


def test_process_frame_returns_expected_structure():
    c = _compressor_with_faces([_face_row(50, 50, 50, 50)])
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)

    det = DetectionInput(
        label="vehicle", confidence=0.90, bbox=BoundingBox(x=50, y=50, w=50, h=50)
    )
    result = c.process_frame(frame, [det], include_preview=True)

    assert result.redaction_enabled is True
    assert result.pii_redacted_count == 1
    assert len(result.rois) == 1
    assert result.rois[0].label == "vehicle"
    assert result.raw_frame_bytes > 0
    assert result.emitted_base64_bytes > 0
    assert result.redacted_preview_base64 is not None


def test_low_confidence_detection_is_dropped():
    c = _compressor_with_faces(None)
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    weak = DetectionInput(
        label="vehicle", confidence=0.10, bbox=BoundingBox(x=0, y=0, w=10, h=10)
    )
    result = c.process_frame(frame, [weak])
    assert result.rois == []
