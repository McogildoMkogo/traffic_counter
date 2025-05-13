import streamlit as st
import json
from pathlib import Path

def load_settings():
    settings_file = Path("utils/settings.json")
    if settings_file.exists():
        with open(settings_file, "r") as f:
            return json.load(f)
    return {
        "detection_confidence": 0.5,
        "tracking_persistence": 30,
        "count_line_position": 0.5,
        "speed_estimation_enabled": True,
        "direction_detection_enabled": True,
        "save_processed_video": False
    }

def save_settings(settings):
    settings_file = Path("utils/settings.json")
    settings_file.parent.mkdir(exist_ok=True)
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=4)

def show_settings():
    st.title("Settings")
    
    settings = load_settings()
    
    st.subheader("Detection Settings")
    detection_conf = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=settings["detection_confidence"],
        step=0.1,
        help="Minimum confidence score for vehicle detection"
    )
    
    tracking_persist = st.number_input(
        "Tracking Persistence (frames)",
        min_value=1,
        max_value=100,
        value=settings["tracking_persistence"],
        help="Number of frames to keep tracking a vehicle after detection is lost"
    )
    
    count_line = st.slider(
        "Count Line Position",
        min_value=0.1,
        max_value=0.9,
        value=settings["count_line_position"],
        step=0.1,
        help="Position of the counting line (as a fraction of video height)"
    )
    
    st.subheader("Feature Settings")
    speed_estimation = st.checkbox(
        "Enable Speed Estimation",
        value=settings["speed_estimation_enabled"],
        help="Calculate approximate vehicle speeds"
    )
    
    direction_detection = st.checkbox(
        "Enable Direction Detection",
        value=settings["direction_detection_enabled"],
        help="Detect and record vehicle movement direction"
    )
    
    save_video = st.checkbox(
        "Save Processed Video",
        value=settings["save_processed_video"],
        help="Save the video with detection overlays"
    )
    
    if st.button("Save Settings"):
        new_settings = {
            "detection_confidence": detection_conf,
            "tracking_persistence": tracking_persist,
            "count_line_position": count_line,
            "speed_estimation_enabled": speed_estimation,
            "direction_detection_enabled": direction_detection,
            "save_processed_video": save_video
        }
        save_settings(new_settings)
        st.success("Settings saved successfully!") 