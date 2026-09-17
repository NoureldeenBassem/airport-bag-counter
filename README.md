# Airport Bag In/Out Counter

Counts bags entering and exiting an airport (via two separate conveyor-belt
videos) using YOLOv8 detection + tracking, then flags a mismatch as a
potential missing bag.

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on `PATH`
(used to make output videos browser-playable).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Upload the entry video and the exit video in the web UI, then click
**Process videos**. The app reports the entry count, exit count, and whether
they match.

## How it works

- `bag_counter/detector.py` loads a pretrained YOLOv8 model (COCO classes:
  suitcase, backpack, handbag).
- `bag_counter/counter.py` runs detection + ByteTrack tracking per video,
  auto-detects the belt's motion direction from the tracked bags themselves
  (so it works whether the belt moves left-right or top-bottom), and counts
  each tracked bag once, the moment its centroid crosses that line — this
  avoids double-counting a bag seen across many frames.
- `app.py` is the Streamlit UI: upload both videos, run both pipelines,
  compare totals. Line orientation can still be forced manually in Settings
  if auto-detect picks wrong for unusual footage.

## Notes

- Works best with a clear single-lane belt shot (one bag-width across the
  frame). A wide-angle shot of a crowded circular carousel can fragment bag
  tracks under occlusion and under/overcount.
- First run downloads the YOLOv8 weights automatically (`yolov8n.pt` by
  default, ~6MB).
