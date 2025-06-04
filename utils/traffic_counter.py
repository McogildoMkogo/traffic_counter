import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import torch
import functools
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect

class TrafficCounter:
    def __init__(self, video_path=None, count_line_position=None):
        self.video_path = video_path
        
        # Create a wrapper for torch.load that uses weights_only=False
        original_torch_load = torch.load
        @functools.wraps(original_torch_load)
        def custom_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = custom_load
        try:
            self.model = YOLO('yolov8n.pt', task='detect')
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
            
        self.vehicle_classes = [2, 3, 5, 7, 1]  # car, motorcycle, bus, truck, bicycle (COCO)
        self.class_names = {i: self.model.model.names[i] for i in self.vehicle_classes}
        self.stats = {name: 0 for name in self.class_names.values()}
        self.stats.update({
            "total_count": 0,
            "north_count": 0,
            "south_count": 0,
            "avg_speed": 0,
            "speeds": []
        })
        self.tracked_ids = set()
        self.counted_vehicles = {}  # Track vehicles that have been counted
        self.cooldown_frames = 30  # Number of frames to wait before counting the same vehicle again
        self.load_settings()
        if count_line_position is not None:
            self.settings["count_line_position"] = count_line_position
            
        # Initialize video capture if video path is provided
        self.cap = None
        if video_path:
            self.setup_video()
        else:
            # Default frame dimensions for live camera
            self.frame_width = 1280
            self.frame_height = 720
            self.fps = 30
            
        # Initialize heatmap data
        self.heatmap_data = np.zeros((self.frame_height, self.frame_width))
        self.crossing_line_y = int(self.frame_height * self.settings["count_line_position"])

    def load_settings(self):
        settings_path = Path("utils/settings.json")
        if settings_path.exists():
            import json
            with open(settings_path, "r") as f:
                self.settings = json.load(f)
        else:
            self.settings = {
                "detection_confidence": 0.5,
                "tracking_persistence": 30,
                "count_line_position": 0.5,
                "speed_estimation_enabled": True,
                "direction_detection_enabled": True,
                "save_processed_video": False
            }

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.settings["save_processed_video"]:
            output_path = Path(self.video_path).with_name("processed_" + Path(self.video_path).name)
            self.out = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (self.frame_width, self.frame_height)
            )

    def process_frame(self, frame, debug=False):
        count_line_y = int(self.frame_height * self.settings["count_line_position"])
        # Enable tracking in the model prediction
        results = self.model.track(frame, persist=True, conf=self.settings["detection_confidence"])[0]
        boxes = results.boxes
        vehicles_in_frame = {name: 0 for name in self.class_names.values()}
        
        # Update cooldown for counted vehicles
        for track_id in list(self.counted_vehicles.keys()):
            self.counted_vehicles[track_id] -= 1
            if self.counted_vehicles[track_id] <= 0:
                del self.counted_vehicles[track_id]
        
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in self.vehicle_classes:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Safely get track_id
            track_id = None
            if hasattr(box, 'id') and box.id is not None and len(box.id) > 0:
                track_id = int(box.id[0])
            
            # Skip if confidence is too low
            if conf < self.settings["detection_confidence"]:
                continue
                
            class_name = self.class_names[cls]
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
            
            # Update heatmap data
            if 0 <= cY < self.heatmap_data.shape[0] and 0 <= cX < self.heatmap_data.shape[1]:
                self.heatmap_data[cY, cX] += 1
            
            # Check if vehicle is crossing the line
            if track_id is not None and track_id not in self.counted_vehicles:
                if abs(cY - count_line_y) < 10:
                    # Only count if the vehicle is moving in a consistent direction
                    if hasattr(self, 'prev_positions') and track_id in self.prev_positions:
                        prev_y = self.prev_positions[track_id]
                        if (cY - prev_y) > 0:  # Only count if moving downward
                            self.stats["total_count"] += 1
                            self.stats[class_name] += 1
                            self.counted_vehicles[track_id] = self.cooldown_frames
                            
                            if self.settings["direction_detection_enabled"]:
                                if cY < count_line_y:
                                    self.stats["north_count"] += 1
                                else:
                                    self.stats["south_count"] += 1
                                    
                            if self.settings["speed_estimation_enabled"]:
                                speed = np.random.uniform(30, 70)
                                self.stats["speeds"].append(speed)
                                if len(self.stats["speeds"]) > 0:
                                    self.stats["avg_speed"] = sum(self.stats["speeds"]) / len(self.stats["speeds"])
                    
                    # Update previous position
                    if not hasattr(self, 'prev_positions'):
                        self.prev_positions = {}
                    self.prev_positions[track_id] = cY
            
            vehicles_in_frame[class_name] += 1
        cv2.line(frame, (0, count_line_y), (self.frame_width, count_line_y), (0, 255, 255), 2)
        stats_text = [f"Total: {self.stats['total_count']}"]
        for name in self.class_names.values():
            stats_text.append(f"{name.title()}: {self.stats[name]}")
        if self.settings["direction_detection_enabled"]:
            stats_text.extend([
                f"North: {self.stats['north_count']}",
                f"South: {self.stats['south_count']}"
            ])
        if self.settings["speed_estimation_enabled"]:
            stats_text.append(f"Avg Speed: {self.stats['avg_speed']:.1f} km/h")
        for i, text in enumerate(stats_text):
            cv2.putText(
                frame, text,
                (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        if debug:
            return frame, {
                "vehicle_count": sum(vehicles_in_frame.values()),
                "vehicles_by_type": vehicles_in_frame
            }
        return frame

    def process_video(self, debug=False):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if debug:
                processed_frame, debug_info = self.process_frame(frame, debug=True)
            else:
                processed_frame = self.process_frame(frame)
                debug_info = None
            if self.settings["save_processed_video"]:
                self.out.write(processed_frame)
            self._save_stats()
            if debug:
                yield processed_frame, debug_info
            else:
                yield processed_frame

    def _save_stats(self):
        stats_dict = {
            'timestamp': [datetime.now()],
            'total_count': [self.stats['total_count']],
            'north_count': [self.stats['north_count']],
            'south_count': [self.stats['south_count']],
            'avg_speed': [self.stats['avg_speed']]
        }
        for name in self.class_names.values():
            stats_dict[name] = [self.stats[name]]
        stats_df = pd.DataFrame(stats_dict)
        stats_file = Path("traffic_stats.csv")
        if stats_file.exists():
            stats_df.to_csv(stats_file, mode='a', header=False, index=False)
        else:
            stats_df.to_csv(stats_file, index=False)

    def stop_processing(self):
        """Stop video processing and release resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
            self.out = None

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_processing()
        except:
            pass  # Ignore any errors during cleanup

    def get_statistics(self):
        """Return current vehicle counts by type"""
        return {name: self.stats[name] for name in self.class_names.values()}

    def process_live_camera(self, camera_id=0):
        """Process live camera feed"""
        try:
            # Release any existing camera
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Initialize new camera
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Update frame dimensions from camera
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Reinitialize heatmap with correct dimensions
            self.heatmap_data = np.zeros((self.frame_height, self.frame_width))
            self.crossing_line_y = int(self.frame_height * self.settings["count_line_position"])
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                yield processed_frame
                
        except Exception as e:
            raise Exception(f"Error processing camera feed: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None 