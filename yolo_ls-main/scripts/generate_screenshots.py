#!/usr/bin/env python3
"""Generate screenshots for mid-term report"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, '/home/ubuntu/yolo_ls')

from src.core.detector import VehicleDetector
from src.core.tracker import ByteTracker
from src.core.collision_risk import CollisionRiskPredictor, RiskLevel
from src.core.feature import FeatureExtractor

# Output directory
OUTPUT_DIR = '/home/ubuntu/yolo_ls/docs/mid-term-report/screenshots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video file
VIDEO_PATH = '/home/ubuntu/下载/2165-155327596_small.mp4'

# Risk level colors
RISK_COLORS = {
    RiskLevel.SAFE: (0, 255, 0),      # Green
    RiskLevel.LOW: (0, 255, 255),     # Yellow
    RiskLevel.MEDIUM: (0, 165, 255),  # Orange
    RiskLevel.HIGH: (0, 0, 255),      # Red
    RiskLevel.CRITICAL: (255, 0, 255) # Purple
}

def draw_detection(frame, detections):
    """Draw detection boxes"""
    result = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(result, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result

def draw_tracking(frame, tracks):
    """Draw tracking boxes with IDs"""
    result = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"ID:{track.track_id} {track.class_name}"
        cv2.putText(result, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return result

def draw_collision_risk(frame, tracks, risks, track_risks):
    """Draw collision risk visualization"""
    result = frame.copy()

    for track in tracks:
        risk_level = track_risks.get(track.track_id, RiskLevel.SAFE)
        color = RISK_COLORS.get(risk_level, (0, 255, 0))

        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        label = f"ID:{track.track_id} [{risk_level.name}]"
        cv2.putText(result, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw risk info
    y_offset = 30
    for risk in risks[:3]:
        text = f"Risk: V{risk.vehicle1_id}<->V{risk.vehicle2_id} TTC:{risk.ttc:.1f}s [{risk.risk_level.name}]"
        cv2.putText(result, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25

    return result

def main():
    print("Initializing components...")
    detector = VehicleDetector(
        model_path='/home/ubuntu/yolo_ls/models/yolo12n_vehicle.pt',
        confidence=0.2,
        device='cuda'
    )
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    risk_predictor = CollisionRiskPredictor(fps=15.0)
    feature_extractor = FeatureExtractor()

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    frame_count = 0
    saved_detection = False
    saved_tracking = False
    saved_collision = False

    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip first few frames
        if frame_count < 30:
            continue

        # Detect
        detections = detector.detect_vehicles(frame)

        # Track
        tracks = tracker.update(detections)

        # Risk prediction - convert Track objects to dict format
        track_dicts = [
            {'track_id': t.track_id, 'bbox': t.bbox, 'class_name': t.class_name}
            for t in tracks
        ]
        risks = risk_predictor.update(track_dicts)
        track_risks = {}
        for risk in risks:
            track_risks[risk.vehicle1_id] = max(
                track_risks.get(risk.vehicle1_id, RiskLevel.SAFE),
                risk.risk_level,
                key=lambda x: x.value
            )
            track_risks[risk.vehicle2_id] = max(
                track_risks.get(risk.vehicle2_id, RiskLevel.SAFE),
                risk.risk_level,
                key=lambda x: x.value
            )

        # Save screenshots when we have enough detections
        if len(detections) >= 3:
            if not saved_detection:
                result = draw_detection(frame, detections)
                path = os.path.join(OUTPUT_DIR, 'detection.png')
                cv2.imwrite(path, result)
                print(f"Saved: {path}")
                saved_detection = True

            if not saved_tracking and len(tracks) >= 3:
                result = draw_tracking(frame, tracks)
                path = os.path.join(OUTPUT_DIR, 'tracking.png')
                cv2.imwrite(path, result)
                print(f"Saved: {path}")
                saved_tracking = True

            if not saved_collision and len(tracks) >= 2:
                result = draw_collision_risk(frame, tracks, risks, track_risks)
                path = os.path.join(OUTPUT_DIR, 'collision_risk.png')
                cv2.imwrite(path, result)
                print(f"Saved: {path}")
                saved_collision = True

        # Stop after saving all screenshots
        if saved_detection and saved_tracking and saved_collision:
            break

        if frame_count > 300:
            print("Reached frame limit")
            break

    cap.release()
    print("Done!")

if __name__ == '__main__':
    main()
