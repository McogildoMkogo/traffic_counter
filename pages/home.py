import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from utils.traffic_counter import TrafficCounter
from utils.live_camera import LiveCameraHandler
from utils.visualization import create_dashboard, load_traffic_data
import pandas as pd
import os

def show_home():
    st.title("Traffic Counter")
    
    # Input source selection
    input_source = st.sidebar.radio("Select Input Source", ["Video File", "Live Camera"])
    
    # Counting line position slider
    count_line_pos = st.sidebar.slider("Counting Line Position (as % of frame height)", min_value=0, max_value=100, value=50, step=1)
    
    if input_source == "Video File":
        # File uploader
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = Path("temp_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize traffic counter with custom count line position
            counter = TrafficCounter(str(video_path))
            counter.crossing_line_y = int(counter.heatmap_data.shape[0] * (count_line_pos / 100.0))
            
            # Create two columns for video and stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Live Feed")
                video_placeholder = st.empty()
                debug_placeholder = st.empty()
                
                # Process video
                frame_skip = 3  # Process every 3rd frame
                for i, (frame, debug_info) in enumerate(counter.process_video(debug=True)):
                    if i % frame_skip != 0:
                        continue
                    # Convert BGR to RGB and resize for smoother display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (640, 360))
                    video_placeholder.image(frame_rgb, channels="RGB")
                    
                    # Display debug info with vehicle types
                    debug_text = f"Vehicles detected in frame: {debug_info['vehicle_count']}\n"
                    for vehicle_type, count in debug_info['vehicles_by_type'].items():
                        debug_text += f"{vehicle_type.replace('_', ' ').title()}: {count}\n"
                    debug_placeholder.info(debug_text)
            
            with col2:
                st.subheader("Statistics")
                stats_placeholder = st.empty()
                
                # Display current statistics
                stats = counter.get_statistics()
                stats_text = f"""
                ### Total Counts
                - Overall Total: {sum(stats.values())}
                
                ### By Vehicle Type
                """
                for vehicle_type, count in stats.items():
                    stats_text += f"- {vehicle_type}: {count}\n"
                
                stats_placeholder.markdown(stats_text)
            
            # Clean up
            counter.stop_processing()
            video_path.unlink()
    
    else:  # Live Camera
        # Initialize traffic counter
        counter = TrafficCounter()
        counter.crossing_line_y = int(counter.heatmap_data.shape[0] * (count_line_pos / 100.0))
        
        # Camera selection
        camera_id = st.sidebar.number_input("Camera ID", min_value=0, max_value=10, value=0, step=1)
        
        # Start/Stop button
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if st.sidebar.button("Start/Stop Camera"):
            if st.session_state.camera_running:
                counter.stop_processing()
                st.session_state.camera_running = False
            else:
                st.session_state.camera_running = True
        
        if st.session_state.camera_running:
            # Create two columns for video and stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Live Camera Feed")
                video_placeholder = st.empty()
                
                try:
                    for processed_frame in counter.process_live_camera(camera_id):
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        # Resize for smoother display
                        frame_rgb = cv2.resize(frame_rgb, (640, 360))
                        video_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Display statistics
                        stats = counter.get_statistics()
                        stats_df = pd.DataFrame(list(stats.items()), columns=['Vehicle Type', 'Count'])
                        st.sidebar.dataframe(stats_df)
                        
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    st.session_state.camera_running = False
            
            with col2:
                st.subheader("Live Statistics")
                stats_placeholder = st.empty()
                
                # Display current statistics
                stats = counter.get_statistics()
                stats_text = f"""
                ### Total Counts
                - Overall Total: {sum(stats.values())}
                
                ### By Vehicle Type
                """
                for vehicle_type, count in stats.items():
                    stats_text += f"- {vehicle_type}: {count}\n"
                
                stats_placeholder.markdown(stats_text)
                
                # Save statistics button
                if st.button("Save Current Statistics"):
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
    
    # Show dashboard if data exists
    st.subheader("Traffic Analysis Dashboard")
    df = load_traffic_data()
    if df is not None and not df.empty:
        fig = create_dashboard(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No traffic data available yet. Upload a video or use live camera to start collecting data.")

    # --- RESET BUTTON ---
    st.markdown('---')
    if st.button('Reset All Data'):
        import os
        import shutil
        # Delete traffic_stats.csv if it exists
        if os.path.exists('traffic_stats.csv'):
            os.remove('traffic_stats.csv')
        # Delete all files in traffic_stats/ directory
        if os.path.exists('traffic_stats'):
            shutil.rmtree('traffic_stats')
        # Delete temp video if it exists
        if os.path.exists('temp_video.mp4'):
            os.remove('temp_video.mp4')
        st.success('All statistics and temporary files have been reset.') 