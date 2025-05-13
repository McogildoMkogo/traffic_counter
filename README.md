# Traffic Counter Application

A real-time traffic counting application using YOLOv8 and Streamlit.

## Features
- Real-time vehicle detection and counting
- Vehicle classification (cars, motorcycles, buses, trucks, bicycles)
- Direction detection (northbound/southbound)
- Speed estimation
- Traffic analytics dashboard

## Deployment
1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Set the main file path as `app.py`
7. Click "Deploy"

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
traffic-counter-pro/
├── app.py              # Main Streamlit application
├── pages/             
│   ├── home.py        # Home page with video processing
│   ├── analytics.py   # Analytics dashboard
│   ├── settings.py    # Settings management
│   └── about.py       # About page
├── utils/
│   ├── traffic_counter.py  # Core traffic counting logic
│   └── visualization.py    # Data visualization utilities
├── static/            # Static assets (CSS, images)
├── traffic_stats/     # Generated statistics
└── requirements.txt   # Project dependencies
```

## Configuration

The application can be configured through the Settings page:

- Detection confidence threshold
- Tracking persistence
- Count line position
- Speed estimation settings
- Direction detection settings
- Video processing options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for object detection
- Streamlit for the web interface
- Supervision for object tracking
- OpenCV for video processing
