# Motorcycle Traffic Counting System

A real-time motorcycle counting system using YOLOv8 object detection and tracking technology. This system can accurately count vehicles crossing predefined lines in video feeds, making it suitable for traffic monitoring and analysis applications.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for accurate motorcycle detection
- **Multi-Object Tracking**: Maintains consistent object IDs across frames
- **Bidirectional Counting**: Separate counters for entry and exit traffic
- **Anti-Double Counting**: Prevents the same object from being counted multiple times
- **Visual Feedback**: Real-time display of counting lines and statistics
- **Console Logging**: Detailed logging of counting events with object IDs
- **Configurable Parameters**: Easy adjustment of detection thresholds and line positions

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows 10/11 (tested)
- GPU recommended for better performance (CUDA-compatible)

### Dependencies
```
opencv-python>=4.5.0
ultralytics>=8.0.0
numpy>=1.21.0
torch>=1.12.0
torchvision>=0.13.0
```

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd Tracking
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv env
   env\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your model and video**
   - Place your trained YOLOv8 model file (e.g., `best.pt`) in the project directory
   - Place your video file in the project directory or update the path in the configuration

## Configuration

Edit the configuration section in `main.py` to customize the system:

```python
# Model Configuration
MODEL_PATH = "best.pt"  # Path to your YOLOv8 model

# Video Configuration
VIDEO_PATH = "your_video.mp4"  # Path to video file or 0 for webcam

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for detections
IOU_THRESHOLD = 0.5         # IoU threshold for NMS

# Counting Line Coordinates
ENTRY_LINE = [(258, 312), (388, 248)]  # Entry line coordinates
EXIT_LINE = [(287, 341), (420, 267)]   # Exit line coordinates
```

### Setting Up Counting Lines

1. **Determine Line Coordinates**: Use an image viewer or video player to identify pixel coordinates
2. **Entry Line**: Define the line where vehicles enter the monitored area
3. **Exit Line**: Define the line where vehicles exit the monitored area
4. **Format**: Each line is defined as `[(x1, y1), (x2, y2)]` where (x1,y1) and (x2,y2) are the start and end points

## Usage

### Basic Usage

```bash
python ver2.py
```

### Controls
- **'q'**: Quit the application
- **ESC**: Alternative quit method

### Output

The system provides:
- **Real-time video display** with detection boxes, counting lines, and statistics
- **Console output** with counting events and object IDs
- **Final statistics** when the program terminates

### Example Output
```
Starting motorcycle counting system...
Entry line: [(258, 312), (388, 248)]
Exit line: [(287, 341), (420, 267)]
Press 'q' to quit

🟢 Motor masuk: 1 (ID: 15)
🔴 Motor keluar: 1 (ID: 23)
🟢 Motor masuk: 2 (ID: 31)

==================================================
FINAL COUNTING RESULTS
==================================================
Total Motor Masuk: 2
Total Motor Keluar: 1
Net Traffic: 1
Total Objects Tracked: 8
Program terminated successfully.
==================================================
```

## How It Works

### Detection and Tracking
1. **Frame Processing**: Each video frame is processed by the YOLOv8 model
2. **Object Detection**: Motorcycles are detected with confidence scores
3. **Object Tracking**: Consistent IDs are assigned to maintain object identity across frames

### Counting Algorithm
1. **Center Point Calculation**: The center point of each detected object's bounding box is calculated
2. **Movement Tracking**: Previous and current positions are compared to determine movement direction
3. **Line Intersection**: Mathematical line intersection algorithm determines if an object crosses a counting line
4. **Anti-Double Counting**: Each object can only be counted once, either as entry or exit

### Mathematical Approach
The system uses parametric line equations to detect intersections:
- Line 1: P = P1 + t(P2-P1)
- Line 2: Q = P3 + u(P4-P3)
- Intersection exists if 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `best.pt` exists in the project directory
   - Check the `MODEL_PATH` configuration

2. **Video not loading**
   - Verify video file path and format
   - Try using absolute paths
   - Ensure video codec is supported

3. **Poor detection accuracy**
   - Adjust `CONFIDENCE_THRESHOLD` (lower for more detections)
   - Retrain model with more diverse data
   - Check lighting and video quality

4. **Incorrect counting**
   - Verify counting line coordinates
   - Adjust line positions based on traffic flow
   - Check for occlusions or overlapping objects

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed for GPU processing
- **Model Size**: Use smaller YOLOv8 models (e.g., yolov8n.pt) for faster processing
- **Resolution**: Reduce video resolution if real-time processing is required
- **Batch Processing**: Process recorded videos for better accuracy

## File Structure

```
Tracking/
├── main.py              # Main application script
├── best.pt              # YOLOv8 trained model
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
├── .gitignore          # Git ignore rules
├── env/                # Virtual environment (optional)
└── your_video.mp4      # Input video file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 implementation
- **OpenCV**: For computer vision utilities
- **PyTorch**: For deep learning framework

## Contact

For questions, issues, or contributions, please contact:
- Email: [your-email@example.com]
- GitHub: [your-github-username]

---

**Note**: This system is designed for motorcycle counting but can be adapted for other vehicle types by retraining the YOLOv8 model with appropriate datasets.