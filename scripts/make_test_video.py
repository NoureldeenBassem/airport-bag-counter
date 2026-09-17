"""Generate a trivial synthetic video for smoke-testing the pipeline (not real bags)."""
import cv2
import numpy as np

out = cv2.VideoWriter("scripts/test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for i in range(40):
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    y = int(20 + i * 5)
    cv2.rectangle(frame, (100, y), (220, y + 60), (60, 60, 60), -1)
    out.write(frame)
out.release()
print("wrote scripts/test.mp4")
