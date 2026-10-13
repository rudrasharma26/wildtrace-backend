"""
Video processor for extracting and preprocessing frames for RoVF inference.
Extracts keyframe sequences, generates thumbnails, and normalizes to model input tensors.
"""

import os
import cv2
import base64
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from io import BytesIO
from PIL import Image

# ImageNet normalization stats used by DINOv2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frame_to_data_url(image_bgr: np.ndarray, quality: int = 75) -> str:
    """Converts a BGR OpenCV image frame to a base64 JPEG data URL for frontend display."""
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pil_img.thumbnail((320, 240))
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_video_frames(
    video_path: str,
    num_frames: int = 5,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts num_frames uniformly from a video file and formats for RoVF model.
    Returns:
        tensor: [1, T, 3, H, W] FloatTensor normalized with ImageNet stats.
        frame_metadata: List of dicts with frame index, timestamp, and thumbnail URL.
        video_metadata: Dict containing duration_seconds, fps, total_frames, width, height.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file. Invalid or unsupported video codec.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # Fallback: count frames by stepping
        count = 0
        while cap.grab():
            count += 1
        total_frames = count
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if total_frames == 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    duration_sec = total_frames / fps if fps > 0 else 0.0

    # Determine uniform sample indices
    if total_frames <= num_frames:
        sample_indices = list(range(total_frames))
        # Pad by repeating last frame if fewer than num_frames
        while len(sample_indices) < num_frames:
            sample_indices.append(sample_indices[-1])
    else:
        sample_indices = [
            int(idx) for idx in np.linspace(0, total_frames - 1, num_frames)
        ]

    processed_frames = []
    frame_metadata = []

    for rank, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Fallback to black frame if read fails
            frame_bgr = np.zeros((height or 224, width or 224, 3), dtype=np.uint8)

        # Generate base64 thumbnail
        thumb_url = frame_to_data_url(frame_bgr)
        timestamp_sec = frame_idx / fps if fps > 0 else 0.0
        m = int(timestamp_sec // 60)
        s = int(timestamp_sec % 60)
        ms = int((timestamp_sec % 1) * 10)
        ts_str = f"{m:02d}:{s:02d}.{ms}"

        frame_metadata.append({
            "frameIndex": rank + 1,
            "originalFrameNumber": frame_idx,
            "timestamp": ts_str,
            "thumbnailUrl": thumb_url,
            "recurrentWeight": 0.0  # Populated after model forward pass
        })

        # Preprocess for model: BGR -> RGB -> Resize -> [0, 1] -> Normalize -> [3, H, W]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose to [C, H, W]
        chw = np.transpose(normalized, (2, 0, 1))
        processed_frames.append(chw)

    cap.release()

    # Stack to [1, T, 3, H, W]
    video_tensor = torch.tensor(np.stack(processed_frames, axis=0), dtype=torch.float32).unsqueeze(0)

    video_meta = {
        "durationSeconds": round(duration_sec, 2),
        "fps": round(fps, 1),
        "totalFrames": total_frames,
        "width": width,
        "height": height
    }

    return video_tensor, frame_metadata, video_meta
