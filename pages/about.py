import streamlit as st

def show_about():
    st.title("About Traffic Counter Pro")
    
    st.markdown("""
    ## Overview
    Traffic Counter Pro is an advanced vehicle detection and counting system that uses 
    state-of-the-art computer vision technology to analyze traffic patterns in real-time.
    
    ### Key Features
    - Real-time vehicle detection and counting
    - Direction-based traffic analysis
    - Speed estimation
    - Advanced analytics and visualization
    - Customizable settings
    - Data export capabilities
    
    ### Technology Stack
    - YOLOv8 for object detection
    - Deep SORT for object tracking
    - Streamlit for the user interface
    - Plotly for data visualization
    - OpenCV for video processing
    
    ### How It Works
    1. Upload a traffic video through the home page
    2. The system processes the video in real-time using YOLOv8
    3. Vehicles are detected and tracked across frames
    4. Direction and speed are calculated for each vehicle
    5. Data is collected and visualized in the analytics section
    
    ### Tips for Best Results
    - Use videos with good lighting conditions
    - Position the camera at an appropriate angle
    - Adjust settings based on your specific needs
    - Regular calibration may be needed for accurate speed estimation
    
    ### Support
    For support or feature requests, please contact the development team.
    
    Version 2.0
    © 2024 All Rights Reserved
    """)
    
    # System information
    st.header("System Information")
    sys_info = st.expander("Show System Information")
    with sys_info:
        import platform
        import torch
        import cv2
        
        st.write(f"- Python Version: {platform.python_version()}")
        st.write(f"- OpenCV Version: {cv2.__version__}")
        st.write(f"- PyTorch Version: {torch.__version__}")
        st.write(f"- CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"- CUDA Version: {torch.version.cuda}")
            st.write(f"- GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Acknowledgments
    st.header("Acknowledgments")
    st.markdown("""
    This application uses several open-source projects:
    - [YOLOv8](https://github.com/ultralytics/ultralytics)
    - [OpenCV](https://opencv.org/)
    - [Streamlit](https://streamlit.io/)
    - [PyTorch](https://pytorch.org/)
    
    Special thanks to the open-source community for their contributions.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Traffic Counter Pro Team") 