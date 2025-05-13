from setuptools import setup, find_packages

setup(
    name="traffic-counter-pro",
    version="2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.3.56",
        "pandas>=1.3.0",
        "plotly>=5.3.0",
        "streamlit>=1.12.0",
        "ultralytics>=8.0.0",
        "supervision>=0.3.0",
        "torch>=1.9.0"
    ],
    python_requires=">=3.8,<3.13",
    author="Traffic Counter Pro Team",
    description="An advanced traffic counting and analysis system using YOLOv8",
    keywords="traffic-counter, yolov8, computer-vision, vehicle-detection",
) 