"""Streamlit UI: upload entry/exit airport videos and check for missing bags."""
import tempfile
from pathlib import Path

import streamlit as st

from bag_counter.counter import process_video

st.set_page_config(page_title="Airport Bag Counter", page_icon="🧳", layout="centered")

st.title("🧳 Airport Bag In/Out Counter")
st.write(
    "Upload the entry (incoming) and exit (outgoing) conveyor-belt videos. "
    "The app counts unique bags crossing the belt in each video and flags a mismatch."
)

col1, col2 = st.columns(2)
with col1:
    entry_file = st.file_uploader("Entry video (bags coming in)", type=["mp4", "avi", "mov", "mkv"], key="entry")
with col2:
    exit_file = st.file_uploader("Exit video (bags going out)", type=["mp4", "avi", "mov", "mkv"], key="exit")

with st.expander("Settings"):
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    orientation_choice = st.selectbox(
        "Counting line orientation",
        ["Auto-detect", "horizontal", "vertical"],
        index=0,
        help="Auto-detect picks the line direction from how the bags actually move in each video.",
    )
    orientation = None if orientation_choice == "Auto-detect" else orientation_choice
    line_ratio = st.slider("Counting line position (fraction of frame)", 0.1, 0.9, 0.5, 0.05)
    conf = st.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
    show_annotated = st.checkbox("Show annotated output videos", value=True)


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


if st.button("Process videos", type="primary", disabled=not (entry_file and exit_file)):
    with st.spinner("Processing entry video..."):
        entry_path = save_upload(entry_file)
        entry_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name if show_annotated else None
        entry_result = process_video(
            entry_path,
            output_path=entry_out,
            model_name=model_name,
            line_ratio=line_ratio,
            orientation=orientation,
            conf=conf,
        )

    with st.spinner("Processing exit video..."):
        exit_path = save_upload(exit_file)
        exit_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name if show_annotated else None
        exit_result = process_video(
            exit_path,
            output_path=exit_out,
            model_name=model_name,
            line_ratio=line_ratio,
            orientation=orientation,
            conf=conf,
        )

    st.subheader("Results")
    r1, r2 = st.columns(2)
    r1.metric("Bags in (entry)", entry_result.count)
    r2.metric("Bags out (exit)", exit_result.count)

    diff = entry_result.count - exit_result.count
    if diff == 0:
        st.success("✅ All bags accounted for — entry and exit counts match.")
    elif diff > 0:
        st.error(f"⚠️ {diff} bag(s) appear to be missing (entry > exit).")
    else:
        st.warning(f"⚠️ Exit count exceeds entry count by {-diff} — check for duplicate counts or extra bags.")

    if show_annotated:
        v1, v2 = st.columns(2)
        with v1:
            st.caption("Entry (annotated)")
            st.video(entry_result.output_path)
        with v2:
            st.caption("Exit (annotated)")
            st.video(exit_result.output_path)
