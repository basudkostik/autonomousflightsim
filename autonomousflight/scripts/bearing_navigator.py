"""
Phase A — Bearing Navigator
============================
High-altitude bearing-based flight toward fire.

FLOW:
  1. Receive bearing angle from AI Tower
  2. Convert bearing to AirSim velocity vector
  3. Fly at high altitude (above canopy) toward the bearing
  4. Continuously capture RGB from drone camera
  5. Run smoke density check on each frame
  6. When smoke density exceeds threshold → transition to Phase B (Plume Tracking)

USAGE:
  Called by MissionController, or standalone:
    python scripts/bearing_navigator.py --bearing 127.0

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Drone already taken off (or this module handles takeoff)
"""

import airsim
import numpy as np
import cv2
import os
import sys
import time
import math
import argparse

# Fix Windows console encoding for emoji output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
# AirSim NED: z is positive downward, so altitude AGL is negative (e.g. -20 ≈ 20 m up).
DEFAULT_CRUISE_ALTITUDE = -20
DEFAULT_SPEED = 6.0                 # m/s cruise speed
SMOKE_DENSITY_THRESHOLD = 0.05     # 5% of image pixels showing smoke → transition
SMOKE_CHECK_INTERVAL = 1.0         # Seconds between smoke density checks
MAX_FLIGHT_TIME = 120              # Safety: max seconds before aborting
VELOCITY_STEP_DURATION = 2.0       # Duration of each velocity command

# HSV range for smoke detection (reuse from smoke_detection.py)
SMOKE_HSV_LOWER = np.array([0, 0, 150])
SMOKE_HSV_UPPER = np.array([180, 50, 255])


class BearingNavigator:
    """
    Phase A: Fly toward a bearing angle at high altitude.
    Transition to Phase B when smoke becomes visible from the drone camera.
    """

    def __init__(self, client, bearing, altitude=DEFAULT_CRUISE_ALTITUDE,
                 speed=DEFAULT_SPEED, smoke_threshold=SMOKE_DENSITY_THRESHOLD):
        """
        Args:
            client: airsim.MultirotorClient (already connected)
            bearing: Target bearing angle in degrees (0=North, 90=East)
            altitude: AirSim Z coordinate (negative = up)
            speed: Cruise speed in m/s
            smoke_threshold: Fraction of pixels that must be smoke to trigger transition
        """
        self.client = client
        self.bearing = bearing
        self.altitude = altitude
        self.speed = speed
        self.smoke_threshold = smoke_threshold

        self.start_time = None
        self.transition_triggered = False
        self.last_smoke_density = 0.0
        self.flight_log = []

        # Output directory for debug images
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "bearing_nav")
        os.makedirs(self.output_dir, exist_ok=True)

    def bearing_to_velocity(self, bearing_deg):
        """
        Convert a bearing angle (degrees) to AirSim velocity vector.

        AirSim coordinate system:
          +X = North, +Y = East, Z = Down (negative = up)
        Bearing: 0°=North, 90°=East, 180°=South, 270°=West

        Returns:
            (vx, vy): Velocity components in m/s
        """
        bearing_rad = math.radians(bearing_deg)
        vx = self.speed * math.cos(bearing_rad)
        vy = self.speed * math.sin(bearing_rad)
        return vx, vy

    def capture_drone_rgb(self):
        """Capture RGB image from the drone's front camera."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            if responses and responses[0].height > 0 and responses[0].width > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img1d.reshape(responses[0].height, responses[0].width, 3)
                return img
        except Exception as e:
            print(f"  ⚠️ Image capture failed: {e}")
        return None

    def check_smoke_density(self, rgb_image):
        """
        Analyze RGB image for smoke presence using HSV filtering.

        Returns:
            float: Smoke density ratio (0.0 to 1.0)
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total_pixels = mask.shape[0] * mask.shape[1]
        smoke_pixels = cv2.countNonZero(mask)
        density = smoke_pixels / total_pixels

        return density

    def should_transition(self):
        """Check if we should transition to Phase B (Plume Tracking)."""
        return self.last_smoke_density >= self.smoke_threshold

    def get_position(self):
        """Get current drone position."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def log_state(self, step, smoke_density):
        """Log current flight state."""
        x, y, z = self.get_position()
        elapsed = time.time() - self.start_time
        entry = {
            "step": step,
            "time": elapsed,
            "x": x, "y": y, "z": z,
            "bearing": self.bearing,
            "smoke_density": smoke_density
        }
        self.flight_log.append(entry)
        return entry

    def fly_toward_bearing(self):
        """
        Main flight loop: fly toward bearing at cruise altitude.
        Blocks until smoke density trigger or timeout.

        Returns:
            dict: Final state with position and reason for transition
        """
        print("\n" + "=" * 55)
        print("  🧭 PHASE A — BEARING NAVIGATION")
        print("=" * 55)
        print(f"  Target bearing: {self.bearing:.1f}°")
        print(f"  Cruise altitude: {abs(self.altitude)}m AGL")
        print(f"  Cruise speed: {self.speed} m/s")
        print(f"  Smoke threshold: {self.smoke_threshold * 100:.1f}%")
        print("-" * 55)

        # Climb to cruise altitude
        print(f"  📡 Climbing to absolute altitude Z: {self.altitude}m...")
        self.client.moveToZAsync(self.altitude, velocity=6) # No join!
        
        # Monitor the climb loop
        climb_start = time.time()
        while time.time() - climb_start < 20: # Give it 20 secs max to reach cruise alt
            current_z = self.get_position()[2]
            print(f"   [Debug Phase A] Climbing... Current Z: {current_z:.2f}m (Target: {self.altitude}m)")
            if current_z <= self.altitude + 2.0: # Threshold error
                print("   ✅ Reached Phase A cruise altitude!")
                break
            time.sleep(1)
        time.sleep(1)

        # Calculate velocity vector
        vx, vy = self.bearing_to_velocity(self.bearing)
        print(f"  🚁 Velocity vector: vx={vx:.2f}, vy={vy:.2f} m/s")

        self.start_time = time.time()
        step = 0

        while True:
            step += 1
            elapsed = time.time() - self.start_time

            # Safety timeout
            if elapsed > MAX_FLIGHT_TIME:
                print(f"\n  ⏱️ Timeout after {MAX_FLIGHT_TIME}s — aborting bearing nav")
                return {
                    "reason": "timeout",
                    "position": self.get_position(),
                    "smoke_density": self.last_smoke_density,
                    "elapsed": elapsed
                }

            # Send velocity command (maintain altitude + heading)
            self.client.moveByVelocityAsync(
                vx, vy, 0,
                duration=VELOCITY_STEP_DURATION,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=self.bearing)
            )

            # Wait for smoke check interval
            time.sleep(SMOKE_CHECK_INTERVAL)

            # Capture and analyze
            rgb = self.capture_drone_rgb()
            if rgb is not None:
                self.last_smoke_density = self.check_smoke_density(rgb)

                # Log state
                entry = self.log_state(step, self.last_smoke_density)
                x, y, z = entry["x"], entry["y"], entry["z"]

                print(f"\r  Step {step:4d} | "
                      f"Pos: ({x:8.1f}, {y:8.1f}) | "
                      f"Alt: {abs(z):5.1f}m | "
                      f"Smoke: {self.last_smoke_density * 100:5.2f}% | "
                      f"Time: {elapsed:5.1f}s",
                      end="", flush=True)

                # Save debug image every 10 steps
                if step % 10 == 0:
                    debug_path = os.path.join(self.output_dir, f"bearing_step_{step}.png")
                    cv2.imwrite(debug_path, rgb)

                # Check transition
                if self.should_transition():
                    print(f"\n\n  🔥 SMOKE DETECTED! Density: {self.last_smoke_density * 100:.2f}%")
                    print(f"  ✅ Transitioning to Phase B — Plume Tracking")

                    # Save transition frame
                    cv2.imwrite(
                        os.path.join(self.output_dir, f"transition_frame_{step}.png"),
                        rgb
                    )

                    # Hover while transitioning
                    self.client.hoverAsync().join()

                    self.transition_triggered = True
                    return {
                        "reason": "smoke_detected",
                        "position": self.get_position(),
                        "smoke_density": self.last_smoke_density,
                        "elapsed": elapsed,
                        "step": step
                    }


def main():
    """Standalone test: fly toward a given bearing."""
    parser = argparse.ArgumentParser(description="Phase A — Bearing Navigator")
    parser.add_argument("--bearing", type=float, default=45.0,
                        help="Target bearing in degrees (default: 45)")
    parser.add_argument("--altitude", type=float, default=DEFAULT_CRUISE_ALTITUDE,
                        help=f"Cruise altitude as negative Z (default: {DEFAULT_CRUISE_ALTITUDE})")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help=f"Cruise speed in m/s (default: {DEFAULT_SPEED})")
    args = parser.parse_args()

    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("🛫 Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    nav = BearingNavigator(client, args.bearing, args.altitude, args.speed)
    result = nav.fly_toward_bearing()

    print(f"\n📊 Result: {result}")
    print(f"📝 Flight log: {len(nav.flight_log)} entries")

    # Land after test
    print("🔽 Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Bearing navigation test complete!")


if __name__ == "__main__":
    main()
