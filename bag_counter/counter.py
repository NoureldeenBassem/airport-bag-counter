"""Line-crossing bag counting for a single video."""
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .detector import BAG_CLASS_IDS, load_model

TRACKER_CONFIG = str(Path(__file__).parent / "bytetrack_bags.yaml")


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


class LineCounter:
    """Counts each tracked bag once, the moment its centroid crosses a line."""

    def __init__(self, line_ratio: float = 0.5, orientation: str = "horizontal"):
        self.line_ratio = line_ratio
        self.orientation = orientation
        self.counted_ids: set[int] = set()
        self._last_side: dict[int, bool] = {}

    def line_pixel(self, frame_shape) -> int:
        h, w = frame_shape[:2]
        return int(h * self.line_ratio) if self.orientation == "horizontal" else int(w * self.line_ratio)

    def update(self, track_id: int, x1: float, y1: float, x2: float, y2: float, frame_shape) -> None:
        line_pos = self.line_pixel(frame_shape)
        centroid = (y1 + y2) / 2 if self.orientation == "horizontal" else (x1 + x2) / 2
        side = centroid > line_pos

        prev_side = self._last_side.get(track_id)
        if prev_side is not None and prev_side != side and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
        self._last_side[track_id] = side

    @property
    def count(self) -> int:
        return len(self.counted_ids)


@dataclass
class VideoResult:
    count: int
    output_path: str | None = None
    frames_processed: int = 0


def _detect_orientation(track_span: dict) -> str:
    """Pick line orientation from how bags actually moved: horizontal motion needs a vertical line."""
    total_dx = sum(abs(s["last"][0] - s["first"][0]) for s in track_span.values())
    total_dy = sum(abs(s["last"][1] - s["first"][1]) for s in track_span.values())
    return "vertical" if total_dx >= total_dy else "horizontal"


def process_video(
    video_path: str,
    output_path: str | None = None,
    model_name: str = "yolov8n.pt",
    line_ratio: float = 0.5,
    orientation: str | None = None,
    conf: float = 0.35,
) -> VideoResult:
    """Run detection+tracking over a video and count unique bags crossing a line.

    orientation=None auto-detects the line direction from each bag's actual movement,
    since a fixed default (e.g. always horizontal) undercounts belts that move the
    other way — a horizontal line never gets crossed by bags moving left-to-right.
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
    # boxes so orientation can be decided from full-video motion before counting starts.
    detections_by_frame: list[list[tuple]] = []
    track_span: dict[int, dict] = {}

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
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls_name = BAG_CLASS_IDS.get(int(cls_id), "bag")
                frame_dets.append((tid, float(x1), float(y1), float(x2), float(y2), cls_name, float(c)))
                span = track_span.setdefault(tid, {"first": (cx, cy)})
                span["last"] = (cx, cy)
        detections_by_frame.append(frame_dets)

    frames_processed = len(detections_by_frame)
    if orientation is None:
        orientation = _detect_orientation(track_span)

    counter = LineCounter(line_ratio=line_ratio, orientation=orientation)
    frame_shape = (height, width)
    line_pos = counter.line_pixel(frame_shape)

    writer = None
    raw_output_path = None
    if output_path:
        raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Pass 2: replay original frames (no re-detection) to finalize crossings and draw.
    cap = cv2.VideoCapture(video_path)
    for frame_dets in detections_by_frame:
        ok, frame = cap.read()
        if not ok:
            break

        for tid, x1, y1, x2, y2, _cls_name, _c in frame_dets:
            counter.update(tid, x1, y1, x2, y2, frame_shape)

        if writer:
            for tid, x1, y1, x2, y2, cls_name, c in frame_dets:
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
            if orientation == "horizontal":
                cv2.line(frame, (0, line_pos), (width, line_pos), (0, 0, 255), 2)
            else:
                cv2.line(frame, (line_pos, 0), (line_pos, height), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Count: {counter.count}",
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

    return VideoResult(count=counter.count, output_path=output_path, frames_processed=frames_processed)
