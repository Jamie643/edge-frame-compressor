"""Privacy-preserving frame compression for edge vision pipelines.

Detects faces, irreversibly redacts them with a solid fill, crops
regions of interest, and reports honest byte accounting for the
payload that actually goes over the wire.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PII_LABELS = frozenset({"face", "person", "license_plate"})


class BoundingBox(BaseModel):
    model_config = ConfigDict(strict=True)

    x: int
    y: int
    w: int
    h: int


class DetectionInput(BaseModel):
    model_config = ConfigDict(strict=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class ROI(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox
    crop_base64: str


class CompressedFrameResult(BaseModel):
    rois: List[ROI]
    redacted_preview_base64: Optional[str] = None
    raw_frame_bytes: int
    emitted_base64_bytes: int
    byte_delta_pct: float
    pii_redacted_count: int
    redaction_enabled: bool


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EdgeFrameCompressor:
    """Edge-vision privacy and bandwidth-optimization layer."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.75,
        yunet_model_path: Optional[str] = None,
        yunet_model_sha256: Optional[str] = None,
        require_redaction: bool = True,
        input_size: Tuple[int, int] = (320, 320),
    ) -> None:
        self.threshold = confidence_threshold
        self.input_size = input_size
        self.yunet_detector = None
        self.redaction_enabled = False

        if yunet_model_path is None:
            if require_redaction:
                raise ValueError(
                    "yunet_model_path is required when require_redaction=True. "
                    "If redaction is handled upstream, pass require_redaction=False."
                )
            logger.warning(
                "EdgeFrameCompressor initialized without redaction. "
                "Emitting frames does NOT scrub PII."
            )
            return

        model_path = Path(yunet_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        if yunet_model_sha256 is None:
            raise ValueError(
                "yunet_model_sha256 must be provided for supply-chain integrity."
            )

        actual = _sha256_of_file(str(model_path))
        if actual != yunet_model_sha256.lower():
            raise RuntimeError(
                "YuNet model hash mismatch — refusing to load.\n"
                f"  expected: {yunet_model_sha256.lower()}\n"
                f"  actual:   {actual}"
            )

        self.yunet_detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
        )
        self.redaction_enabled = True

    @staticmethod
    def _clamp_bbox(
        bbox: BoundingBox, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        if bbox.w <= 0 or bbox.h <= 0:
            return None

        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

        if x >= img_w or y >= img_h or (x + w) <= 0 or (y + h) <= 0:
            return None

        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0

        w = min(img_w - x, w)
        h = min(img_h - y, h)

        if w <= 0 or h <= 0:
            return None

        return (x, y, w, h)

    def redact_pii(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return a redacted copy of `frame` and the count of redactions."""
        redacted = frame.copy()
        redacted_count = 0

        if not self.redaction_enabled or self.yunet_detector is None:
            return redacted, redacted_count

        h_img, w_img = frame.shape[:2]
        self.yunet_detector.setInputSize((w_img, h_img))
        _, faces = self.yunet_detector.detect(frame)

        if faces is None:
            return redacted, redacted_count

        for face in faces:
            fx, fy, fw, fh = (int(round(v)) for v in face[:4])
            clamped = self._clamp_bbox(
                BoundingBox(x=fx, y=fy, w=fw, h=fh), w_img, h_img
            )
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            cv2.rectangle(
                redacted, (cx, cy), (cx + cw, cy + ch), (0, 0, 0), -1
            )
            redacted_count += 1

        return redacted, redacted_count

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionInput],
        *,
        include_preview: bool = False,
    ) -> CompressedFrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty image frame provided.")

        h_img, w_img = frame.shape[:2]

        ok, raw_encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("Failed to encode source frame as JPEG.")
        raw_bytes = int(raw_encoded.nbytes)

        redacted_frame, redacted_count = self.redact_pii(frame)

        extracted_rois: List[ROI] = []
        total_emitted_b64_bytes = 0

        for det in detections:
            if det.confidence < self.threshold:
                continue

            if det.label.lower() in PII_LABELS and not self.redaction_enabled:
                logger.error(
                    "Emitting ROI labeled %r but redaction is DISABLED.",
                    det.label,
                )

            clamped = self._clamp_bbox(det.bbox, w_img, h_img)
            if clamped is None:
                continue

            x, y, w, h = clamped
            crop = redacted_frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            ok, crop_encoded = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                continue

            b64_str = base64.b64encode(crop_encoded.tobytes()).decode("ascii")
            total_emitted_b64_bytes += len(b64_str.encode("ascii"))

            extracted_rois.append(
                ROI(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    crop_base64=b64_str,
                )
            )

        preview_b64: Optional[str] = None
        if include_preview:
            ok, prev_encoded = cv2.imencode(
                ".jpg", redacted_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if ok:
                preview_b64 = base64.b64encode(
                    prev_encoded.tobytes()
                ).decode("ascii")
                total_emitted_b64_bytes += len(preview_b64.encode("ascii"))

        byte_delta_pct = (
            round(((raw_bytes - total_emitted_b64_bytes) / raw_bytes) * 100, 2)
            if raw_bytes > 0
            else 0.0
        )

        return CompressedFrameResult(
            rois=extracted_rois,
            redacted_preview_base64=preview_b64,
            raw_frame_bytes=raw_bytes,
            emitted_base64_bytes=total_emitted_b64_bytes,
            byte_delta_pct=byte_delta_pct,
            pii_redacted_count=redacted_count,
            redaction_enabled=self.redaction_enabled,
        )
