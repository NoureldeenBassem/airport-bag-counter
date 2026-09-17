# Airport Bag In/Out Counter

Counts bags entering and exiting an airport (via two separate conveyor-belt
videos) using YOLOv8 detection + tracking, then flags a mismatch as a
potential missing bag.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Upload the entry video and the exit video in the web UI, adjust the
counting-line position/orientation if needed for your footage, then click
**Process videos**. The app reports the entry count, exit count, and whether
they match.

## How it works

- `bag_counter/detector.py` loads a pretrained YOLOv8 model (COCO classes:
  suitcase, backpack, handbag).
- `bag_counter/counter.py` runs detection + ByteTrack tracking per video and
  counts each tracked bag once, the moment its centroid crosses a
  configurable line — this avoids double-counting a bag seen across many
  frames.
- `app.py` is the Streamlit UI: upload both videos, run both pipelines,
  compare totals.

## Notes

- No sample footage was available while building this — test with any
  overhead/side view of a baggage conveyor belt. Adjust the line position
  and orientation in the Settings panel to match where bags cross in your
  footage.
- First run downloads the YOLOv8 weights automatically (`yolov8n.pt` by
  default, ~6MB).
