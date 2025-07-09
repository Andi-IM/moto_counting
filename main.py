#!/usr/bin/env python3
"""
Motorcycle Traffic Counting System using YOLOv8 Object Detection and Tracking

This script implements a real-time motorcycle counting system that tracks vehicles
crossing predefined lines in a video feed. It uses YOLOv8 for object detection and
tracking, with custom line intersection algorithms for accurate counting.

Features:
- Real-time object detection and tracking
- Bidirectional counting (entry and exit)
- Prevents double counting of the same object
- Visual feedback with counting lines and statistics
- Console logging of counting events

Version: 1.0
"""

import cv2
from ultralytics import YOLO
import numpy as np

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Model Configuration
# Replace with your custom model path (e.g., 'yolov8n.pt' or custom trained model)
MODEL_PATH = "best.pt"

# Video Configuration
# Path to input video file or use 0 for webcam
VIDEO_PATH = "D02 20250303105720-smaller.mp4"

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for detections
IOU_THRESHOLD = 0.5         # IoU threshold for NMS

# Counting Line Coordinates
# Format: [(x1, y1), (x2, y2)] - defines start and end points of counting lines
ENTRY_LINE = [(258, 312), (388, 248)]  # Green line for vehicles entering
EXIT_LINE = [(287, 341), (420, 267)]   # Red line for vehicles exiting

# Visual Configuration
LINE_THICKNESS = 3
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Counting variables
motor_masuk = 0   # Counter for vehicles entering
motor_keluar = 0  # Counter for vehicles exiting

# Tracking data structures
previous_positions = {}  # Store previous center positions of tracked objects
crossed_entry = set()    # Set of track IDs that crossed entry line
crossed_exit = set()     # Set of track IDs that crossed exit line
counted_objects = {}     # Dictionary to prevent double counting {track_id: 'entry'/'exit'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def line_intersection(p1, p2, p3, p4):
    """
    Check if two line segments intersect using parametric line equations.
    
    This function determines whether the line segment from p1 to p2 intersects
    with the line segment from p3 to p4. Uses the parametric form of line equations
    and checks if the intersection point lies within both line segments.
    
    Args:
        p1 (tuple): Start point of first line segment (x1, y1)
        p2 (tuple): End point of first line segment (x2, y2)
        p3 (tuple): Start point of second line segment (x3, y3)
        p4 (tuple): End point of second line segment (x4, y4)
    
    Returns:
        bool: True if line segments intersect, False otherwise
    
    Mathematical approach:
        - Uses parametric equations: P = P1 + t(P2-P1) and Q = P3 + u(P4-P3)
        - Solves for parameters t and u
        - Intersection exists if 0 <= t <= 1 and 0 <= u <= 1
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate denominator for parametric equations
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    # Check if lines are parallel (denominator close to zero)
    if abs(denom) < 1e-10:
        return False
    
    # Calculate parameters t and u
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    # Check if intersection point lies within both line segments
    return 0 <= t <= 1 and 0 <= u <= 1

def draw_line(frame, line, color, thickness=LINE_THICKNESS):
    """
    Draw a line on the given frame.
    
    Args:
        frame (numpy.ndarray): The image frame to draw on
        line (list): List containing two points [(x1, y1), (x2, y2)]
        color (tuple): BGR color tuple (B, G, R)
        thickness (int): Line thickness in pixels
    """
    cv2.line(frame, line[0], line[1], color, thickness)

def get_center_point(box):
    """
    Calculate the center point of a bounding box.
    
    Args:
        box (numpy.ndarray or list): Bounding box coordinates [x1, y1, x2, y2]
                                    where (x1, y1) is top-left and (x2, y2) is bottom-right
    
    Returns:
        tuple: Center point coordinates (center_x, center_y) as integers
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

print("Starting motorcycle counting system...")
print(f"Entry line: {ENTRY_LINE}")
print(f"Exit line: {EXIT_LINE}")
print("Press 'q' to quit\n")

while cap.isOpened():
    # Read frame from video
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame")
        break

    # Perform object detection and tracking
    # persist=True ensures object IDs are maintained across frames
    results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True)

    # Get annotated frame with detection boxes and labels
    annotated_frame = results[0].plot()
    
    # Draw counting lines on the frame
    draw_line(annotated_frame, ENTRY_LINE, (0, 255, 0))  # Green line for entry
    draw_line(annotated_frame, EXIT_LINE, (0, 0, 255))   # Red line for exit
    
    # Process detections for counting
    # Only proceed if there are valid detections with tracking IDs
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Extract bounding boxes and tracking IDs from detection results
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Tracking IDs
        
        # Process each detected object
        for box, track_id in zip(boxes, track_ids):
            # Calculate center point of current bounding box
            current_center = get_center_point(box)
            
            # Check if we have a previous position for this tracked object
            # This is necessary to determine movement direction and line crossing
            if track_id in previous_positions:
                prev_center = previous_positions[track_id]
                
                # Check if object trajectory crosses the entry line
                if line_intersection(prev_center, current_center, ENTRY_LINE[0], ENTRY_LINE[1]):
                    # Ensure object hasn't been counted before (prevents double counting)
                    if track_id not in crossed_entry and track_id not in counted_objects:
                        motor_masuk += 1
                        crossed_entry.add(track_id)
                        counted_objects[track_id] = 'entry'
                        print(f"🟢 Motor masuk: {motor_masuk} (ID: {track_id})")
                
                # Check if object trajectory crosses the exit line
                if line_intersection(prev_center, current_center, EXIT_LINE[0], EXIT_LINE[1]):
                    # Ensure object hasn't been counted before (prevents double counting)
                    if track_id not in crossed_exit and track_id not in counted_objects:
                        motor_keluar += 1
                        crossed_exit.add(track_id)
                        counted_objects[track_id] = 'exit'
                        print(f"🔴 Motor keluar: {motor_keluar} (ID: {track_id})")
            
            # Update previous position for next frame comparison
            previous_positions[track_id] = current_center
    
    # Display counting statistics on frame
    # Show entry counter in green
    cv2.putText(annotated_frame, f"Motor Masuk: {motor_masuk}", (10, 30), 
                TEXT_FONT, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)
    # Show exit counter in red
    cv2.putText(annotated_frame, f"Motor Keluar: {motor_keluar}", (10, 70), 
                TEXT_FONT, TEXT_SCALE, (0, 0, 255), TEXT_THICKNESS)
    
    # Add descriptive labels for counting lines
    cv2.putText(annotated_frame, "MASUK", (ENTRY_LINE[0][0]-20, ENTRY_LINE[0][1]-10), 
                TEXT_FONT, 0.6, (0, 255, 0), TEXT_THICKNESS)
    cv2.putText(annotated_frame, "KELUAR", (EXIT_LINE[0][0]-20, EXIT_LINE[0][1]-10), 
                TEXT_FONT, 0.6, (0, 0, 255), TEXT_THICKNESS)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Motorcycle Tracking and Counting System", annotated_frame)

    # Check for quit command (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================================
# CLEANUP AND FINAL RESULTS
# ============================================================================

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Print final statistics
print("\n" + "="*50)
print("FINAL COUNTING RESULTS")
print("="*50)
print(f"Total Motor Masuk: {motor_masuk}")
print(f"Total Motor Keluar: {motor_keluar}")
print(f"Net Traffic: {motor_masuk - motor_keluar}")
print(f"Total Objects Tracked: {len(previous_positions)}")
print("Program terminated successfully.")
print("="*50)
