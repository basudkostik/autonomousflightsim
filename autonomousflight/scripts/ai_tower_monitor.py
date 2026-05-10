"""
AI Tower - Continuous Monitoring System
========================================
Real-time fire/smoke detection using Background Subtraction.

FLOW:
  1. Connect to AirSim
  2. Every 5 seconds, capture a frame from FixedCamera1
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
import datetime


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
CAPTURE_INTERVAL = 5           # Seconds between each photo
CHANGE_THRESHOLD = 25          # Pixel intensity difference to count as "changed"
CHANGE_AREA_MIN = 2000         # Minimum changed pixel count to trigger alarm
CHANGE_AREA_PER_CONTOUR_MIN = 200  # Each individual contour must be at least this large (filters noise blobs)
CAMERA_FOV = 120               # Must match settings.json FOV_Degrees
TOWER_YAW = 230                # Must match settings.json Yaw

# Drone mission start position (AirSim meters)
DRONE_START_X = 0.0
DRONE_START_Y = 0.0
DRONE_START_Z = 0.0

# ── HSV Ranges for validation ────────────────────────────
# Smoke: Low saturation, medium-to-high value (Grey/White)
SMOKE_HSV_LOWER = np.array([0, 0, 150])
SMOKE_HSV_UPPER = np.array([180, 50, 255])

# Fire: High saturation, high value (Orange/Red)
FIRE_HSV_LOWER = np.array([0, 80, 150])
FIRE_HSV_UPPER = np.array([35, 255, 255])

# ── Static Exclusion Zones ───────────────────────────────
# Pixel rectangles (x1, y1, x2, y2) that are ALWAYS ignored.
# ⚠️  The tower camera is fixed, so river/lake pixels never move.
# Add rectangles here that cover water surfaces visible from the tower.
# Look at the saved BASELINE_*.png to measure river pixel coordinates.
#
# Example: river is roughly pixels x=1150–1920, y=200–450 in 1920×1080
EXCLUSION_ZONES = [
    (1150, 180, 1920, 480),   # River / water reflection zone (adjust to your scene)
]

# ── Multi-frame Confirmation ─────────────────────────────
# The alarm only triggers after this many CONSECUTIVE detections.
# River ripples are random each frame → won't survive N consecutive checks.
# Smoke/fire grows → triggers the same region in every frame.
CONSECUTIVE_DETECTIONS_REQUIRED = 2
# ─────────────────────────────────────────────────────────

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


def build_exclusion_mask(image_shape):
    """
    Build a binary mask (same size as the camera image) where EXCLUSION_ZONES
    are blacked out (0). Everything outside the zones stays 255.
    Pre-built once and reused every frame for performance.
    """
    h, w = image_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        mask[y1:y2, x1:x2] = 0
    return mask


def compare_frames(prev_frame, curr_frame, threshold=25, exclusion_mask=None):
    """
    Compare two frames using:
      1. Motion detection  (grayscale absdiff)
      2. HSV color filter  (only smoke/fire colors count)
      3. Exclusion zones   (river / water pixels permanently ignored)
    Returns: (final_mask, contours, total_area, change_center)
    """
    # ── 1. MOTION DETECTION ──────────────────────────────────
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # ── 2. HSV COLOR FILTER ──────────────────────────────────
    # River water reflecting the sky looks grey/white — identical to smoke.
    # We still run this filter to bias toward smoke-colored regions.
    hsv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    smoke_mask = cv2.inRange(hsv_curr, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER)
    fire_mask  = cv2.inRange(hsv_curr, FIRE_HSV_LOWER,  FIRE_HSV_UPPER)
    color_mask = cv2.bitwise_or(smoke_mask, fire_mask)

    # ── 3. EXCLUSION ZONES ───────────────────────────────────
    # Black out river / lake pixels BEFORE any further analysis.
    # Because the tower camera is fixed, water is always in the same pixels.
    if exclusion_mask is not None:
        motion_mask = cv2.bitwise_and(motion_mask, exclusion_mask)
        color_mask  = cv2.bitwise_and(color_mask,  exclusion_mask)

    # ── 4. COMBINE ───────────────────────────────────────────
    final_mask = cv2.bitwise_and(motion_mask, color_mask)

    # Dilate to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # Find contours and filter tiny blobs
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= CHANGE_AREA_PER_CONTOUR_MIN]

    total_change_area = sum(cv2.contourArea(c) for c in contours)

    # Centroid of the largest contour
    change_center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            change_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return final_mask, contours, total_change_area, change_center


def calculate_bearing(pixel_x, image_width, camera_fov, tower_yaw=0):
    """Convert a pixel X position to a world bearing angle."""
    relative_offset = (pixel_x / image_width) - 0.5
    angle_offset = relative_offset * camera_fov
    bearing = (tower_yaw + angle_offset) % 360
    return bearing


def dispatch_drone(client, bearing):
    """
    Dispatch the drone on a self-chaining 2-phase mission:
      A → BearingNavigator  (cruise + descent toward bearing)
      C → PPO Training      (on-site training, triggered by A)

    PlumeTracker is no longer used.
    ai_tower_monitor only starts Phase A.
    BearingNavigator handles descent and chains to PPO directly.
    """
    print("\n" + "=" * 55)
    print("  🚁 DISPATCHING DRONE — CHAIN MISSION START")
    print(f"  🎯 Target bearing: {bearing:.2f}°")
    print("=" * 55)

    # ── Detect fire position BEFORE takeoff ────────────────
    # Must happen while drone is on the ground — querying AirSim
    # scene objects during flight can cause API conflicts.
    from bearing_navigator import BearingNavigator, detect_fire_position
    print("  🔍 Detecting fire position from scene...")
    fire_x, fire_y = detect_fire_position(client)

    # ── Clean Reset ─────────────────────────────────────────
    # client.reset() is safe here — we're at the START of the mission,
    # not inside the PPO training loop. This clears any stale drone state
    # (previous crash, collision, already airborne) that would cause
    # takeoffAsync().join() to hang indefinitely.
    print("  🔄 Resetting drone to clean state...")
    client.reset()
    time.sleep(1)

    # ── Takeoff ──────────────────────────────────────────────
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    print("  🛫 Taking off...")
    client.takeoffAsync()  # No .join() — monitor manually with timeout

    # Monitor takeoff with timeout (prevents hanging forever)
    takeoff_start = time.time()
    while time.time() - takeoff_start < 15:  # 15s max for takeoff
        z = client.getMultirotorState().kinematics_estimated.position.z_val
        print(f"   [Takeoff] Z: {z:.2f}m", flush=True)
        if z < -0.5:  # At least 0.5m above ground
            print("   ✅ Takeoff successful!")
            break
        time.sleep(1)
    else:
        print("   ⚠️ Takeoff timeout — proceeding anyway")
    time.sleep(1)

    # ── Phase A: BearingNavigator (chains to PPO directly) ──
    nav = BearingNavigator(client, bearing, fire_x=fire_x, fire_y=fire_y)
    final_result = nav.fly_toward_bearing()

    # ── Mission complete — land ───────────────────────────────
    fire_confirmed = final_result.get("fire_confirmed", False) if isinstance(final_result, dict) else False
    print("\n" + "=" * 55)
    print(f"  {'🔥✅ FIRE CONFIRMED!' if fire_confirmed else '❌ Fire not confirmed.'}")
    print("  🔽 Returning and landing...")
    print("=" * 55)
    try:
        client.moveToZAsync(-10, 5).join()
        client.moveToPositionAsync(0, 0, -10, 8).join()
        client.landAsync().join()
    except Exception:
        pass
    client.armDisarm(False)
    client.enableApiControl(False)
    print("  ✅ Landed safely. Mission complete.")
    return fire_confirmed


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
    ts_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("📸 Capturing baseline frame...")
    prev_frame = capture_tower_image(client)
    if prev_frame is None:
        print("❌ Failed to capture baseline. Check UE4 and settings.json.")
        return

    baseline_path = os.path.join(output_dir, f"BASELINE_{ts_start}.png")
    cv2.imwrite(baseline_path, prev_frame)
    print(f"   Baseline saved → {baseline_path}")
    print(f"   Image size: {prev_frame.shape}")
    print(f"   Monitoring every {CAPTURE_INTERVAL} seconds...")
    print(f"   Change threshold: {CHANGE_THRESHOLD} | Min area: {CHANGE_AREA_MIN}")
    print("-" * 55)

    frame_count = 0
    alarm_triggered = False
    consecutive_detections = 0   # counts consecutive frames with change >= CHANGE_AREA_MIN

    # Build exclusion mask once — river pixels are always at the same location
    exc_mask = build_exclusion_mask(prev_frame.shape)
    if EXCLUSION_ZONES:
        print(f"   🚫 Exclusion zones active: {len(EXCLUSION_ZONES)} zone(s) masked out")
        for z in EXCLUSION_ZONES:
            print(f"      {z}")

    # ──────────────────────────────────────────────
    # 3. Continuous monitoring loop
    # ──────────────────────────────────────────────
    try:
        while not alarm_triggered:
            # Wait for next capture
            print(f"\n⏳ Waiting {CAPTURE_INTERVAL}s for next capture...", end="", flush=True)
            time.sleep(CAPTURE_INTERVAL)
            frame_count += 1

            ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Capture new frame
            curr_frame = capture_tower_image(client)
            if curr_frame is None:
                print(" ❌ Capture failed, retrying...")
                continue

            # Save every monitoring frame with a timestamped, numbered name
            frame_path = os.path.join(output_dir, f"MONITOR_frame{frame_count:04d}_{ts_now}.png")
            cv2.imwrite(frame_path, curr_frame)

            # Compare with previous (clear) frame
            diff_mask, contours, change_area, change_center = compare_frames(
                prev_frame, curr_frame, CHANGE_THRESHOLD, exc_mask
            )

            print(f"\r📸 Frame #{frame_count:04d} [{ts_now}] | Changed pixels: {change_area:,} / {CHANGE_AREA_MIN} | Consecutive: {consecutive_detections}/{CONSECUTIVE_DETECTIONS_REQUIRED}", end="")

            if change_area >= CHANGE_AREA_MIN and change_center is not None:
                consecutive_detections += 1
                print(f" | 🔸 Hit {consecutive_detections}/{CONSECUTIVE_DETECTIONS_REQUIRED}", end="")

                if consecutive_detections < CONSECUTIVE_DETECTIONS_REQUIRED:
                    # Not enough consecutive hits yet — don't reset prev_frame so
                    # next frame compares against the same clean reference
                    continue

            if change_area >= CHANGE_AREA_MIN and change_center is not None and consecutive_detections >= CONSECUTIVE_DETECTIONS_REQUIRED:
                # ══════════════════════════════════════════
                # 🚨 ALARM TRIGGERED!
                # ══════════════════════════════════════════
                cx, cy = change_center
                h, w = curr_frame.shape[:2]
                bearing = calculate_bearing(cx, w, CAMERA_FOV, TOWER_YAW)

                print(f"\n\n🚨🔥 ALARM! Change detected at pixel ({cx}, {cy})")
                print(f"   Changed area : {change_area:,} pixels")
                print(f"   Bearing      : {bearing:.2f}°")
                print(f"   Saving 2 pre-dispatch photos...")

                # ── Photo 1: The BEFORE frame (last clear frame used as reference) ──
                before_path = os.path.join(
                    output_dir,
                    f"ALARM_{frame_count:04d}_1_BEFORE_{ts_now}.png"
                )
                cv2.imwrite(before_path, prev_frame)
                print(f"   📷 BEFORE (reference) → {os.path.basename(before_path)}")

                # ── Photo 2: The AFTER frame (the one that triggered the alarm) ──
                after_path = os.path.join(
                    output_dir,
                    f"ALARM_{frame_count:04d}_2_AFTER_{ts_now}.png"
                )
                cv2.imwrite(after_path, curr_frame)
                print(f"   📷 AFTER  (triggered)  → {os.path.basename(after_path)}")

                # ── Photo 3: Pixel-diff mask ──
                diff_path = os.path.join(
                    output_dir,
                    f"ALARM_{frame_count:04d}_3_DIFF_{ts_now}.png"
                )
                cv2.imwrite(diff_path, diff_mask)
                print(f"   📷 DIFF   (mask)       → {os.path.basename(diff_path)}")

                # ── Photo 4: Annotated detection with contours + dot + bearing text ──
                viz = curr_frame.copy()
                cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
                cv2.circle(viz, (cx, cy), 15, (0, 0, 255), -1)
                cv2.putText(
                    viz,
                    f"FIRE DETECTED!  Bearing: {bearing:.1f} deg  |  Area: {change_area:,} px",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                )
                cv2.putText(
                    viz,
                    f"Frame #{frame_count:04d}  |  {ts_now}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )
                annotated_path = os.path.join(
                    output_dir,
                    f"ALARM_{frame_count:04d}_4_ANNOTATED_{ts_now}.png"
                )
                cv2.imwrite(annotated_path, viz)
                print(f"   📷 ANNOTATED           → {os.path.basename(annotated_path)}")

                # Dispatch the drone!
                alarm_triggered = True
                dispatch_drone(client, bearing)

            else:
                consecutive_detections = 0   # reset streak on any clear frame
                print(f" | ✅ Clear", end="")
                # Update previous frame only when scene is clear (no false baselines)
                prev_frame = curr_frame.copy()

    except KeyboardInterrupt:
        print("\n\n⛔ Monitoring stopped by user.")

    print("", flush=True)  # Force newline to prevent progress bar \r from corrupting output
    print("\n" + "=" * 55)
    print(f"  Monitoring ended. Total frames captured: {frame_count}")
    print("=" * 55)


if __name__ == "__main__":
    main()
