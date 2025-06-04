import cv2
import streamlit as st
import time
from pathlib import Path
import pandas as pd
import os

class LiveCameraHandler:
    def __init__(self, traffic_counter):
        self.traffic_counter = traffic_counter
        self.is_running = False
        self.camera_id = 0
        self.frame_count = 0

    def start_camera(self, camera_id=0):
        """Start the live camera feed"""
        self.camera_id = camera_id
        self.is_running = True
        self.frame_count = 0

    def stop_camera(self):
        """Stop the live camera feed"""
        self.is_running = False

    def process_camera_feed(self):
        """Process the live camera feed"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                processed_frame = self.traffic_counter.process_frame(frame, self.frame_count)
                yield processed_frame

                # Add a small delay to control frame rate
                time.sleep(1/30)  # Assuming 30 FPS

        finally:
            cap.release()

    def save_statistics(self):
        """Save the current statistics"""
        if self.traffic_counter.stats_log:
            # Create a stats directory if it doesn't exist
            stats_dir = Path("traffic_stats")
            stats_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csv_path = stats_dir / f"live_camera_stats_{timestamp}.csv"
            
            # Convert stats log to DataFrame
            df = pd.DataFrame(self.traffic_counter.stats_log)
            df.to_csv(csv_path, index=False)
            return str(csv_path)
        return None 