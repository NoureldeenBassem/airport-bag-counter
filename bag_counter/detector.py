"""YOLO model loading for bag detection."""
from functools import lru_cache

from ultralytics import YOLO

# COCO class ids for luggage-like objects.
BAG_CLASS_IDS = {24: "backpack", 26: "handbag", 28: "suitcase"}


@lru_cache(maxsize=1)
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    return YOLO(model_name)
