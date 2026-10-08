# edge-frame-compressor

Privacy-preserving frame extraction for AI vision pipelines.

## What it does

- Detects faces using YuNet (ONNX, CPU-friendly).
- Redacts them irreversibly with a solid black fill — not a blur.
- Crops high-confidence regions of interest for downstream use.
- Reports honest byte accounting including base64 inflation.

## What it does NOT do (yet)

- License plate detection. Face-only for now.
- Video streams. Single frames only.
- GPU acceleration. CPU only.

## Install

    uv sync --extra dev

## Usage

    import cv2
    from edge_frame_compressor import (
        EdgeFrameCompressor, DetectionInput, BoundingBox,
    )

    compressor = EdgeFrameCompressor(
        yunet_model_path="models/yunet.onnx",
        yunet_model_sha256="<see models/README.md>",
    )

    frame = cv2.imread("photo.jpg")
    detections = [
        DetectionInput(
            label="vehicle", confidence=0.92,
            bbox=BoundingBox(x=100, y=100, w=200, h=200),
        ),
    ]
    result = compressor.process_frame(frame, detections, include_preview=True)
    print(result.byte_delta_pct, "percent saved")

## Byte accounting

`byte_delta_pct` is computed as:

    (raw_frame_bytes - emitted_base64_bytes) / raw_frame_bytes * 100

It is *negative* when the emitted payload exceeds the raw frame. This
happens when ROIs collectively cover most of the frame, or when
`include_preview=True` adds a second full-frame JPEG.

Real reductions come from emitting many small ROIs instead of one large
frame — a 4K camera frame with a car and a pedestrian crop yields a
large positive number. Emitting a single ROI the size of the frame
yields a negative number, and the library reports that honestly rather
than clamping it.

# YuNet face detection model

File: `yunet.onnx`
Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Version: 2023mar

SHA-256:

    <8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4>

Any code that loads this model MUST verify the hash before use.
See `EdgeFrameCompressor.__init__` for the check.

## Trust model

- YuNet model hash is verified at startup. A swapped model file
  causes a hard failure, not silent misbehavior.
- If redaction is disabled, the result object records
  `redaction_enabled=False` and PII-labeled ROIs log at ERROR.
- Redaction uses solid fill, not Gaussian blur. Blur is
  reversible under deconvolution attacks; a black rectangle is not.
  
## Status

v0.1.0 — internal. Not yet published.
