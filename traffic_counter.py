import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TrafficCounter:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        # COCO dataset class IDs for vehicles (0-based indexing)
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck'
        }
        self.counts = {class_id: 0 for class_id in self.vehicle_classes.keys()}
        self.tracked_objects = {}
        self.crossing_line_y = None
        self.speed_estimates = {}
        self.fps = 30  # Default FPS, will be updated from video
        
        # Initialize statistics logging
        self.stats_log = []
        self.heatmap_data = np.zeros((720, 1280))  # Default size, will be updated

    def calculate_speed(self, track_id, x, y, frame_time):
        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = []
        
        self.tracked_objects[track_id].append((x, y, frame_time))
        
        # Keep only last 10 positions for each object
        if len(self.tracked_objects[track_id]) > 10:
            self.tracked_objects[track_id].pop(0)
            
        # Calculate speed if we have at least 2 positions
        if len(self.tracked_objects[track_id]) >= 2:
            pos1 = self.tracked_objects[track_id][-2]
            pos2 = self.tracked_objects[track_id][-1]
            
            # Calculate distance in pixels
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            time_diff = pos2[2] - pos1[2]
            
            speed = (distance * 0.1 * 3.6) / time_diff if time_diff > 0 else 0
            return min(speed, 200)  # Cap speed at 200 km/h to avoid unrealistic values
        return 0

    def process_frame(self, frame, frame_number):
        if self.crossing_line_y is None:
            self.crossing_line_y = frame.shape[0] // 2
            self.heatmap_data = np.zeros(frame.shape[:2])

        # Draw counting line
        cv2.line(frame, (0, self.crossing_line_y), (frame.shape[1], self.crossing_line_y),
                 (0, 255, 255), 2)

        # Run YOLOv8 detection with tracking
        try:
            results = self.model.track(frame, persist=True, classes=list(self.vehicle_classes.keys()))
            
            if results and results[0].boxes is not None and hasattr(results[0].boxes, 'data'):
                detections = results[0].boxes.data.cpu().numpy()
                
                current_time = time.time()
                
                for detection in detections:
                    if len(detection) >= 7:  # Ensure we have tracking ID
                        x1, y1, x2, y2, conf, class_id, track_id = detection[:7]
                        class_id = int(class_id)
                        
                        # Skip if not a vehicle class we're interested in
                        if class_id not in self.vehicle_classes:
                            continue
                            
                        if conf > 0.3:  # Confidence threshold
                            track_id = int(track_id)
                            
                            # Calculate center point
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Ensure coordinates are within bounds
                            center_y = min(center_y, self.heatmap_data.shape[0]-1)
                            center_x = min(center_x, self.heatmap_data.shape[1]-1)
                            
                            # Update heatmap
                            self.heatmap_data[center_y, center_x] += 1
                            
                            # Calculate speed
                            speed = self.calculate_speed(track_id, center_x, center_y, current_time)
                            
                            # Check if vehicle crossed the line
                            if track_id not in self.speed_estimates:
                                self.speed_estimates[track_id] = {"prev_y": None, "counted": False}
                            
                            prev_y = self.speed_estimates[track_id]["prev_y"]
                            if prev_y is not None:
                                # Check if vehicle crossed the line from top to bottom
                                if prev_y < self.crossing_line_y <= center_y and not self.speed_estimates[track_id]["counted"]:
                                    self.counts[class_id] += 1
                                    self.speed_estimates[track_id]["counted"] = True
                                    
                                    # Log the crossing event
                                    self.stats_log.append({
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'frame': frame_number,
                                        'vehicle_type': self.vehicle_classes[class_id],
                                        'speed': speed
                                    })
                            
                            self.speed_estimates[track_id]["prev_y"] = center_y
                            
                            # Draw bounding box and information
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label = f"{self.vehicle_classes[class_id]}: {speed:.1f} km/h"
                            cv2.putText(frame, label, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
        
        return frame

    def get_statistics(self):
        return {self.vehicle_classes[k]: v for k, v in self.counts.items()}
    
    def save_statistics(self):
        # Create a stats directory if it doesn't exist
        stats_dir = Path("traffic_stats")
        stats_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = stats_dir / f"traffic_stats_{timestamp}.csv"
        
        # Convert stats log to DataFrame
        if self.stats_log:
            df = pd.DataFrame(self.stats_log)
            df.to_csv(csv_path, index=False)
            return str(csv_path)
        return None

    def get_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.heatmap_data, cmap='YlOrRd')
        plt.title('Vehicle Detection Heatmap')
        return plt

def main():
    st.title("Enhanced Traffic Counter using YOLOv8")
    
    # Sidebar options
    st.sidebar.title("Settings")
    show_heatmap = st.sidebar.checkbox("Show Heatmap", False)
    save_stats = st.sidebar.checkbox("Save Statistics", True)
    
    # Initialize traffic counter
    counter = TrafficCounter()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = Path("temp_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Video processing
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                st.error("Error: Could not open video file")
                return
                
            counter.fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create placeholders
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            heatmap_placeholder = st.empty() if show_heatmap else None
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process frame
                processed_frame = counter.process_frame(frame, frame_count)
                
                # Convert BGR to RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(processed_frame)
                
                # Display statistics
                stats = counter.get_statistics()
                stats_df = pd.DataFrame(list(stats.items()), columns=['Vehicle Type', 'Count'])
                stats_placeholder.dataframe(stats_df)
                
                # Update heatmap if enabled
                if show_heatmap and frame_count % 30 == 0:  # Update every 30 frames
                    heatmap_fig = counter.get_heatmap()
                    heatmap_placeholder.pyplot(heatmap_fig)
                    plt.close()
            
            # Save statistics if enabled
            if save_stats:
                csv_path = counter.save_statistics()
                if csv_path:
                    st.success(f"Statistics saved to: {csv_path}")
                    
                    # Add download button for CSV
                    with open(csv_path, 'rb') as file:
                        st.download_button(
                            label="Download Statistics CSV",
                            data=file,
                            file_name=os.path.basename(csv_path),
                            mime="text/csv"
                        )
            
            cap.release()
            temp_path.unlink()  # Delete temporary file
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if 'cap' in locals():
                cap.release()
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    main() 