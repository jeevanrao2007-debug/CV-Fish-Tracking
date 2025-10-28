#!/usr/bin/env python3
"""
track_fish.py
Detect and track fish in a pond video using background subtraction and contour tracking.
"""

import cv2
import numpy as np
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Track fish in pond video")
    p.add_argument("--video", "-v", required=True, help="Path to pond video file or webcam index")
    p.add_argument("--out", "-o", default="fish_tracked.mp4", help="Output video path")
    p.add_argument("--min-area", type=int, default=800, help="Minimum contour area to count as fish")
    p.add_argument("--debug", action="store_true", help="Show live video feed")
    return p.parse_args()

def main():
    args = parse_args()

    # Open video or webcam
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open video source:", args.video)
        sys.exit(1)

    # Prepare background subtractor (good for moving objects)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Get frame info for saving output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    fish_count = 0
    frame_num = 0

    print("🎥 Tracking fish... press [Q] to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Apply background subtraction
        fgMask = backSub.apply(frame)

        # Remove shadows and noise
        _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fish_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            fish_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Fish {fish_count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Add overlay text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Fish detected: {fish_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

        if args.debug:
            cv2.imshow("Fish Tracking", frame)
            cv2.imshow("Foreground Mask", fgMask)
            if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q')]:
                break

    cap.release()
    out.release()
    if args.debug:
        cv2.destroyAllWindows()

    print(f"✅ Tracking finished — saved output video to: {args.out}")

if __name__ == "__main__":
    main()
