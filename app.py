import streamlit as st
import cv2
import tempfile
import numpy as np
from src.processor import VideoProcessor
from src.detector import ensure_model
import os

st.set_page_config(page_title="AI Toll Booth", layout="wide")

st.title("🚗 Automatic Toll Booth Monitor")
st.markdown("Automated vehicle detection and traffic analysis using YOLOv8.")

# Sidebar Controls
st.sidebar.header("Settings")
model_size = st.sidebar.selectbox("Model Size", ["yolov8n", "yolov8s", "yolov8m"])
conf_level = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
line_pos = st.sidebar.slider("Detection Line Position", 0.1, 0.9, 0.6)

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Ensure model is available under ./models/<model>.pt. If it's missing,
    # show a UI button to download it into the project's models/ folder.
    model_filename = f"{model_size}.pt"
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)

    if not os.path.exists(model_path):
        # Silently ensure the model exists in models/ (no interactive button).
        # We show a spinner while attempting the download. If it fails,
        # fall back to letting Ultralytics resolve the model from its cache.
        with st.spinner(f"Ensuring model {model_filename} is available..."):
            try:
                ensured = ensure_model(model_filename)
                # If ensure_model returned a path, use it
                if ensured and os.path.exists(ensured):
                    model_path = ensured
                else:
                    # keep using model_path (Ultralytics cache will be used)
                    pass
            except Exception as e:
                st.error(f"Model download failed, attempting to continue using Ultralytics cache: {e}")
                # Continue; YOLO will use cache or remote name
                model_path = model_filename

    processor = VideoProcessor(model_path, line_y_ratio=line_pos)
    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()
    
    with col2:
        st.subheader("Live Statistics")
        ent_metric = st.metric("Vehicles Entering", 0)
        ext_metric = st.metric("Vehicles Leaving", 0)

    stop_btn = st.button("Stop Processing")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, counts = processor.process_frame(frame, conf_level)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update UI
        video_placeholder.image(frame_rgb, use_container_width=True)
        ent_metric.metric("Vehicles Entering", counts["entering"])
        ext_metric.metric("Vehicles Leaving", counts["leaving"])

    cap.release()
    st.success("Processing Complete!")