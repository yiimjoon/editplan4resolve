# OpenCV Integration

## Overview
VideoForge uses OpenCV for lightweight video analysis utilities (frame sampling,
scene detection, and quality checks).

## VideoAnalyzer
`VideoForge.adapters.video_analyzer.VideoAnalyzer` provides:
- `get_video_info(video_path)` for width/height/fps/duration
- `extract_frame_at_time(video_path, timestamp)` for frame extraction

## Install
OpenCV is listed in `pyproject.toml`:
```
opencv-python-headless>=4.8.0,<5.0.0
```

## Tests
Run tests if `test_media/Aroll/sample1.mp4` is available:
```
pytest tests/test_opencv_install.py tests/test_video_analyzer.py
```
