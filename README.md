# edge-frame-compressor

Privacy-preserving frame extraction for AI vision pipelines.

## What it does

- Detects faces in an image using YuNet (ONNX, CPU-friendly).
- Redacts them irreversibly with a solid black fill — not a blur.
- Crops high-confidence regions of interest for downstream use.
- Reports honest byte accounting for the payload that actually
  goes over the wire (including base64 inflation).

## What it does NOT do (yet)

- License plate detection. Face-only for now.
- Video streams. Single frames only.
- GPU acceleration. CPU only.

## Install

    uv sync

## Usage

    import cv2
    from edge_frame_compressor import (
        EdgeFrameCompressor, DetectionInput, BoundingBox,
    )

    compressor = EdgeFrameCompressor(
        yunet_model_path="models/yunet.onnx",
        yunet_model_sha256="<your hash here>",
    )

    frame = cv2.imread("photo.jpg")
    detections = [
        DetectionInput(
            label="vehicle",
            confidence=0.92,
            bbox=BoundingBox(x=100, y=100, w=200, h=200),
        ),
    ]
    result = compressor.process_frame(frame, detections, include_preview=True)
    print(result.byte_delta_pct, "percent saved")

## Trust model

- The YuNet model hash is verified at startup. A swapped model file
  causes a hard failure, not silent misbehavior.
- If redaction is disabled (`require_redaction=False`), the result
  object records `redaction_enabled=False` and every PII-labeled
  ROI triggers an ERROR log. Callers cannot silently ship
  unredacted frames.
- Redaction uses solid fill, not Gaussian blur. Blur is reversible
  under deconvolution attacks; a black rectangle is not.

## Tests

    uv run pytest

## Status

v0.1.0 — internal. Not yet published.
