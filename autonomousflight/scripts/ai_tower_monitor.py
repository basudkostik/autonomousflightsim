"""
AI Tower - Continuous Monitoring System
========================================
Real-time fire/smoke detection using Background Subtraction.

FLOW:
  1. Connect to AirSim
  2. Every 10 seconds, capture a frame from FixedCamera1
  3. Compare with the previous frame (pixel difference)
  4. If change exceeds threshold → ALARM → Spawn & dispatch drone
  5. Calculate bearing angle toward the detected change

PREREQUISITES:
  1. settings.json with ExternalCameras > FixedCamera1
  2. UE4 running with Play pressed
  3. Fire/smoke will appear during monitoring

USAGE:
  python scripts/ai_tower_monitor.py

Press Ctrl+C to stop monitoring.
"""

import airsim
import numpy as np
import cv2
import os
import time
import math


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
CAPTURE_INTERVAL = 10          # Seconds between each photo
CHANGE_THRESHOLD = 25          # Pixel intensity difference to count as "changed"
CHANGE_AREA_MIN = 500          # Minimum changed pixel count to trigger alarm
CAMERA_FOV = 120               # Must match settings.json FOV_Degrees
TOWER_YAW = 0                  # Must match settings.json Yaw

# Drone mission start position (AirSim meters)
DRONE_START_X = 0.0
DRONE_START_Y = 0.0
DRONE_START_Z = -2.0


def capture_tower_image(client):
    """Capture a single frame from the tower camera."""
    responses = client.simGetImages([
        airsim.ImageRequest(
            camera_name="FixedCamera1",
            image_type=airsim.ImageType.Scene,
            pixels_as_float=False,
            compress=False
        )
    ], external=True)

    if responses and responses[0].height > 0 and responses[0].width > 0:
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(response.height, response.width, 3)
        return img
    return None


def compare_frames(prev_frame, curr_frame, threshold=25):
    """
    Compare two frames using absolute difference.
    Returns: (change_mask, change_area, change_center)
    """
    # Convert both to grayscale
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)

    # Absolute difference
    diff = cv2.absdiff(gray_prev, gray_curr)

    # Threshold the difference
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of changed regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_change_area = sum(cv2.contourArea(c) for c in contours)

    # Find center of the largest change
    change_center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            change_center = (cx, cy)

    return thresh, contours, total_change_area, change_center


def calculate_bearing(pixel_x, image_width, camera_fov, tower_yaw=0):
    """Convert a pixel X position to a world bearing angle."""
    relative_offset = (pixel_x / image_width) - 0.5
    angle_offset = relative_offset * camera_fov
    bearing = (tower_yaw + angle_offset) % 360
    return bearing


def dispatch_drone(client, bearing):
    """
    Dispatch drone on full 3-phase fire verification mission.

    Phases:
      A — Bearing Navigation (high altitude)
      B — Plume Tracking (mid altitude, descent)
      C — PPO Navigation (under canopy, fire confirmation)
    """
    print("\n" + "=" * 50)
    print("🚁 DISPATCHING DRONE — FULL MISSION!")
    print(f"🎯 Target bearing: {bearing:.2f}°")
    print("=" * 50)

    try:
        from mission_controller import MissionController
        controller = MissionController(client)
        report = controller.run_mission(bearing)
        return report.get("fire_confirmed", False)
    except ImportError:
        # Fallback if mission_controller not available
        print("⚠️ MissionController not found, using basic dispatch...")
        client.enableApiControl(True)
        client.armDisarm(True)
        print("🛫 Taking off...")
        client.takeoffAsync().join()
        time.sleep(2)
        client.moveToZAsync(-20, 5).join()
        time.sleep(1)
        print(f"✅ Drone airborne at bearing {bearing:.2f}°")
        return True


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # 1. Connect to AirSim
    # ──────────────────────────────────────────────
    print("=" * 55)
    print("  🏔️  AI TOWER - CONTINUOUS MONITORING SYSTEM")
    print("=" * 55)
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!\n")

    # ──────────────────────────────────────────────
    # 2. Capture baseline frame
    # ──────────────────────────────────────────────
    print("📸 Capturing baseline frame...")
    prev_frame = capture_tower_image(client)
    if prev_frame is None:
        print("❌ Failed to capture baseline. Check UE4 and settings.json.")
        return

    cv2.imwrite(os.path.join(output_dir, "baseline.png"), prev_frame)
    print(f"   Baseline saved. Image size: {prev_frame.shape}")
    print(f"   Monitoring every {CAPTURE_INTERVAL} seconds...")
    print(f"   Change threshold: {CHANGE_THRESHOLD} | Min area: {CHANGE_AREA_MIN}")
    print("-" * 55)

    frame_count = 0
    alarm_triggered = False

    # ──────────────────────────────────────────────
    # 3. Continuous monitoring loop
    # ──────────────────────────────────────────────
    try:
        while not alarm_triggered:
            # Wait for next capture
            print(f"\n⏳ Waiting {CAPTURE_INTERVAL}s for next capture...", end="", flush=True)
            time.sleep(CAPTURE_INTERVAL)
            frame_count += 1

            # Capture new frame
            curr_frame = capture_tower_image(client)
            if curr_frame is None:
                print(" ❌ Capture failed, retrying...")
                continue

            # Save the captured frame
            frame_path = os.path.join(output_dir, f"monitor_{frame_count}.png")
            cv2.imwrite(frame_path, curr_frame)

            # Compare with previous frame
            diff_mask, contours, change_area, change_center = compare_frames(
                prev_frame, curr_frame, CHANGE_THRESHOLD
            )

            print(f"\r📸 Frame #{frame_count} | Changed pixels: {change_area:,}", end="")

            if change_area >= CHANGE_AREA_MIN and change_center is not None:
                # ══════════════════════════════════
                # 🚨 ALARM TRIGGERED!
                # ══════════════════════════════════
                cx, cy = change_center
                h, w = curr_frame.shape[:2]
                bearing = calculate_bearing(cx, w, CAMERA_FOV, TOWER_YAW)

                print(f"\n\n🚨🔥 ALARM! Change detected at pixel ({cx}, {cy})")
                print(f"   Changed area: {change_area:,} pixels")
                print(f"   Bearing: {bearing:.2f}°")

                # Save detection visualization
                viz = curr_frame.copy()
                cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
                cv2.circle(viz, (cx, cy), 15, (0, 0, 255), -1)
                cv2.putText(viz, f"FIRE DETECTED! Bearing: {bearing:.1f} deg",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Save diff image
                diff_path = os.path.join(output_dir, f"alarm_diff_{frame_count}.png")
                cv2.imwrite(diff_path, diff_mask)

                # Save annotated detection
                detect_path = os.path.join(output_dir, f"alarm_detection_{frame_count}.png")
                cv2.imwrite(detect_path, viz)

                print(f"   📝 Diff image: {diff_path}")
                print(f"   📝 Detection image: {detect_path}")

                # Dispatch the drone!
                alarm_triggered = True
                dispatch_drone(client, bearing)

            else:
                print(f" | Status: ✅ Clear", end="")
                # Update previous frame for next comparison
                prev_frame = curr_frame.copy()

    except KeyboardInterrupt:
        print("\n\n⛔ Monitoring stopped by user.")

    print("\n" + "=" * 55)
    print(f"  Monitoring ended. Total frames captured: {frame_count}")
    print("=" * 55)


if __name__ == "__main__":
    main()
