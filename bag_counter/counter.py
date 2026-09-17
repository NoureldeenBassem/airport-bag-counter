"""Bag counting for a single belt video: count every distinct bag that appears."""
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from .detector import BAG_CLASS_IDS, load_model

TRACKER_CONFIG = str(Path(__file__).parent / "bytetrack_bags.yaml")
MIN_TRACK_FRAMES = 10  # drop brief spurious detections (glare, reflections) as noise


def _transcode_to_h264(src: str, dst: str) -> None:
    """Re-encode to H.264 so the file plays in browsers (OpenCV's mp4v codec often doesn't)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(src, dst)
        return
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
        check=True,
        capture_output=True,
    )
    Path(src).unlink(missing_ok=True)


@dataclass
class VideoResult:
    count: int
    output_path: str | None = None
    frames_processed: int = 0


def process_video(
    video_path: str,
    output_path: str | None = None,
    model_name: str = "yolov8n.pt",
    conf: float = 0.35,
    min_track_frames: int = MIN_TRACK_FRAMES,
) -> VideoResult:
    """Count every distinct bag tracked on the belt, no line-crossing required.

    A bag counts once it's been tracked for at least `min_track_frames` frames —
    that's enough to appear on the belt at all, and filters out brief spurious
    detections (glare, reflections) without needing the bag to cross any
    particular point in the frame.
    """
    model = load_model(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    class_ids = list(BAG_CLASS_IDS.keys())
    results = model.track(
        source=video_path,
        classes=class_ids,
        conf=conf,
        persist=True,
        stream=True,
        tracker=TRACKER_CONFIG,
        verbose=False,
    )

    # Pass 1: detect+track (the expensive step), buffer only the lightweight per-frame
    # boxes so noise tracks can be filtered out before the final count is fixed.
    detections_by_frame: list[list[tuple]] = []
    track_frame_counts: dict[int, int] = {}

    for result in results:
        frame_dets = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, cls_id, c in zip(xyxy, ids, clss, confs):
                tid = int(tid)
                cls_name = BAG_CLASS_IDS.get(int(cls_id), "bag")
                frame_dets.append((tid, float(x1), float(y1), float(x2), float(y2), cls_name, float(c)))
                track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1
        detections_by_frame.append(frame_dets)

    frames_processed = len(detections_by_frame)
    confirmed_ids = {tid for tid, n in track_frame_counts.items() if n >= min_track_frames}

    writer = None
    raw_output_path = None
    if output_path:
        raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Pass 2: replay original frames (no re-detection) to draw and tally the running count.
    cap = cv2.VideoCapture(video_path)
    seen_so_far: set[int] = set()
    for frame_dets in detections_by_frame:
        ok, frame = cap.read()
        if not ok:
            break

        for tid, *_rest in frame_dets:
            if tid in confirmed_ids:
                seen_so_far.add(tid)

        if writer:
            for tid, x1, y1, x2, y2, cls_name, c in frame_dets:
                if tid not in confirmed_ids:
                    continue
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"id:{tid} {cls_name} {c:.2f}",
                    (p1[0], max(p1[1] - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.putText(
                frame,
                f"Count: {len(seen_so_far)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        _transcode_to_h264(raw_output_path, output_path)

    return VideoResult(count=len(confirmed_ids), output_path=output_path, frames_processed=frames_processed)
