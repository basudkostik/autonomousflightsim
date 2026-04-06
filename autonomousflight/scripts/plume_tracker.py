"""
Phase B — Plume Tracker
========================
Mid-altitude smoke plume tracking to find fire source.

ALGORITHM:
  1. Capture RGB from drone camera
  2. Detect smoke region using HSV filtering
  3. Calculate smoke centroid + gradient direction
  4. The fire is in the OPPOSITE direction of smoke flow
  5. Fly INTO the smoke source (against the wind)
  6. Gradually descend as smoke density increases

FLOW:
  Phase A (BearingNavigator) → TRANSITION → Phase B (PlumeTracker) → Phase C (PPO)

USAGE:
  Called by MissionController, or standalone:
    python scripts/plume_tracker.py

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Drone already airborne near smoke area
"""

import airsim
import numpy as np
import cv2
import os
import sys
import time
import math

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
START_ALTITUDE = -25               # Starting altitude (negative = up)
MIN_ALTITUDE = -8                  # Lowest altitude before PPO handoff
DESCENT_RATE = 0.3                 # m per step to descend
TRACKING_SPEED = 3.0               # m/s, slower than bearing nav
STEP_DURATION = 1.5                # Duration of each velocity command
MAX_TRACKING_TIME = 90             # Safety timeout (seconds)

# Smoke density threshold to trigger PPO handoff
PPO_HANDOFF_DENSITY = 0.15        # 15% smoke coverage → hand to PPO

# HSV range for smoke detection
SMOKE_HSV_LOWER = np.array([0, 0, 150])
SMOKE_HSV_UPPER = np.array([180, 50, 255])

# Fire/ember HSV range (reddish-orange glow)
FIRE_HSV_LOWER = np.array([0, 100, 200])
FIRE_HSV_UPPER = np.array([25, 255, 255])


class PlumeTracker:
    """
    Phase B: Follow smoke plume backwards to find fire source.
    Gradually descend while tracking smoke gradient.
    """

    def __init__(self, client, start_altitude=START_ALTITUDE,
                 descent_rate=DESCENT_RATE, speed=TRACKING_SPEED):
        """
        Args:
            client: airsim.MultirotorClient (already connected)
            start_altitude: Starting Z coordinate (negative = up)
            descent_rate: Meters to descend per tracking step
            speed: Tracking speed in m/s
        """
        self.client = client
        self.current_altitude = start_altitude
        self.descent_rate = descent_rate
        self.speed = speed

        self.transition_to_ppo = False
        self.fire_detected = False
        self.last_smoke_density = 0.0
        self.flight_log = []
        self.prev_smoke_mask = None

        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "plume_track")
        os.makedirs(self.output_dir, exist_ok=True)

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

    def detect_smoke_mask(self, rgb_image):
        """
        Create a binary smoke mask from an RGB image.

        Returns:
            numpy.ndarray: Binary mask (0 or 255) of smoke pixels
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER)

        # Morphological cleaning
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def detect_fire_mask(self, rgb_image):
        """Detect fire/ember pixels (reddish-orange glow)."""
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_smoke_gradient(self, rgb_image):
        """
        Determine the direction smoke is flowing by analyzing density
        across image quadrants.

        The fire is in the OPPOSITE direction of the smoke flow.

        Returns:
            (direction_x, direction_y): Normalized direction vector pointing
                                        FROM sparse smoke TOWARD dense smoke
                                        (i.e., toward the fire source)
        """
        smoke_mask = self.detect_smoke_mask(rgb_image)
        h, w = smoke_mask.shape

        # Divide image into a 3x3 grid for finer gradient estimation
        grid_rows, grid_cols = 3, 3
        cell_h = h // grid_rows
        cell_w = w // grid_cols

        density_map = np.zeros((grid_rows, grid_cols))
        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = smoke_mask[r * cell_h:(r + 1) * cell_h,
                                  c * cell_w:(c + 1) * cell_w]
                density_map[r, c] = cv2.countNonZero(cell) / (cell_h * cell_w)

        # Calculate weighted centroid of smoke density
        total_density = density_map.sum()
        if total_density < 0.001:
            # No smoke visible — continue current heading
            return 0.0, 0.0

        # Weighted average position (in grid coordinates)
        weighted_col = 0.0
        weighted_row = 0.0
        for r in range(grid_rows):
            for c in range(grid_cols):
                weighted_col += c * density_map[r, c]
                weighted_row += r * density_map[r, c]
        weighted_col /= total_density
        weighted_row /= total_density

        # Convert to direction: center of grid is (1, 1)
        # Positive direction_x = smoke is denser on the right → fire is right
        # Positive direction_y = smoke is denser at bottom → fire is below/forward
        center_col = (grid_cols - 1) / 2.0
        center_row = (grid_rows - 1) / 2.0

        direction_x = weighted_col - center_col  # Left(-) to Right(+)
        direction_y = weighted_row - center_row  # Top(-) to Bottom(+)

        # Normalize
        magnitude = math.sqrt(direction_x ** 2 + direction_y ** 2)
        if magnitude > 0.01:
            direction_x /= magnitude
            direction_y /= magnitude

        # Store for next frame comparison
        self.prev_smoke_mask = smoke_mask

        return direction_x, direction_y

    def estimate_fire_direction(self, smoke_dir_x, smoke_dir_y):
        """
        Convert smoke gradient direction to AirSim velocity.

        The gradient points toward denser smoke → toward the fire.
        Map image directions to AirSim world:
          Image right (dir_x > 0) → AirSim +Y
          Image bottom (dir_y > 0) → AirSim +X (forward)

        Returns:
            (vx, vy): Velocity command to fly toward fire
        """
        # Map image space to AirSim velocity space
        vx = smoke_dir_y * self.speed   # Image vertical → forward/back
        vy = smoke_dir_x * self.speed   # Image horizontal → left/right

        return vx, vy

    def get_smoke_density(self, rgb_image):
        """Calculate overall smoke density in the image."""
        mask = self.detect_smoke_mask(rgb_image)
        total = mask.shape[0] * mask.shape[1]
        smoke = cv2.countNonZero(mask)
        return smoke / total

    def check_fire_visible(self, rgb_image):
        """Check if fire/embers are directly visible in the image."""
        fire_mask = self.detect_fire_mask(rgb_image)
        fire_pixels = cv2.countNonZero(fire_mask)
        return fire_pixels > 200  # At least 200 fire-colored pixels

    def should_transition_to_ppo(self):
        """
        Check if conditions are met to hand off to PPO agent:
          - Altitude is low enough (below canopy)
          - OR smoke density is very high (close to fire)
          - OR fire is directly visible
        """
        altitude_low = self.current_altitude >= MIN_ALTITUDE
        density_high = self.last_smoke_density >= PPO_HANDOFF_DENSITY
        return altitude_low or density_high or self.fire_detected

    def get_position(self):
        """Get current drone position."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def log_state(self, step, smoke_density, dir_x, dir_y):
        """Log current tracking state."""
        x, y, z = self.get_position()
        elapsed = time.time() - self.start_time
        entry = {
            "step": step, "time": elapsed,
            "x": x, "y": y, "z": z,
            "altitude": self.current_altitude,
            "smoke_density": smoke_density,
            "gradient_x": dir_x, "gradient_y": dir_y,
            "fire_visible": self.fire_detected
        }
        self.flight_log.append(entry)
        return entry

    def track_plume(self):
        """
        Main plume tracking loop.
        Follow smoke gradient toward fire source while descending.

        Returns:
            dict: Final state with position and reason for transition
        """
        print("\n" + "=" * 55)
        print("  🌫️ PHASE B — PLUME TRACKING")
        print("=" * 55)
        print(f"  Start altitude: {abs(self.current_altitude)}m AGL")
        print(f"  Min altitude (PPO handoff): {abs(MIN_ALTITUDE)}m AGL")
        print(f"  Tracking speed: {self.speed} m/s")
        print(f"  Descent rate: {self.descent_rate} m/step")
        print("-" * 55)

        # Ensure we are at the starting altitude
        self.client.moveToZAsync(self.current_altitude, 3).join()
        time.sleep(1)

        self.start_time = time.time()
        step = 0

        while True:
            step += 1
            elapsed = time.time() - self.start_time

            # Safety timeout
            if elapsed > MAX_TRACKING_TIME:
                print(f"\n  ⏱️ Timeout after {MAX_TRACKING_TIME}s — aborting plume tracking")
                self.client.hoverAsync().join()
                return {
                    "reason": "timeout",
                    "position": self.get_position(),
                    "smoke_density": self.last_smoke_density,
                    "elapsed": elapsed
                }

            # Capture frame
            rgb = self.capture_drone_rgb()
            if rgb is None:
                time.sleep(0.5)
                continue

            # Analyze smoke gradient
            dir_x, dir_y = self.detect_smoke_gradient(rgb)
            self.last_smoke_density = self.get_smoke_density(rgb)
            self.fire_detected = self.check_fire_visible(rgb)

            # Convert to velocity
            vx, vy = self.estimate_fire_direction(dir_x, dir_y)

            # If no smoke visible, keep going forward slowly
            if abs(dir_x) < 0.01 and abs(dir_y) < 0.01:
                vx = self.speed * 0.5  # Drift forward
                vy = 0.0

            # Gradually descend
            if self.last_smoke_density > 0.02:
                self.current_altitude = min(
                    self.current_altitude + self.descent_rate,
                    MIN_ALTITUDE
                )

            # Send velocity command with altitude
            self.client.moveByVelocityAsync(
                vx, vy, 0,
                duration=STEP_DURATION
            )
            # Maintain altitude
            self.client.moveToZAsync(self.current_altitude, 2)

            # Log
            entry = self.log_state(step, self.last_smoke_density, dir_x, dir_y)
            x, y, z = entry["x"], entry["y"], entry["z"]

            fire_str = "🔥" if self.fire_detected else "  "
            print(f"\r  Step {step:4d} | "
                  f"Pos: ({x:8.1f}, {y:8.1f}) | "
                  f"Alt: {abs(z):5.1f}m | "
                  f"Smoke: {self.last_smoke_density * 100:5.2f}% | "
                  f"Dir: ({dir_x:+.2f}, {dir_y:+.2f}) {fire_str}",
                  end="", flush=True)

            # Save debug image periodically
            if step % 5 == 0:
                debug_img = rgb.copy()
                h, w = debug_img.shape[:2]
                # Draw gradient arrow
                cx, cy = w // 2, h // 2
                ex = int(cx + dir_x * 80)
                ey = int(cy + dir_y * 80)
                cv2.arrowedLine(debug_img, (cx, cy), (ex, ey),
                                (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(debug_img,
                            f"Smoke: {self.last_smoke_density * 100:.1f}% Alt: {abs(self.current_altitude):.0f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(
                    os.path.join(self.output_dir, f"plume_step_{step}.png"),
                    debug_img
                )

            # Check PPO handoff
            if self.should_transition_to_ppo():
                reason_parts = []
                if self.current_altitude >= MIN_ALTITUDE:
                    reason_parts.append("altitude_low")
                if self.last_smoke_density >= PPO_HANDOFF_DENSITY:
                    reason_parts.append("high_density")
                if self.fire_detected:
                    reason_parts.append("fire_visible")
                reason = "+".join(reason_parts)

                print(f"\n\n  ✅ PPO HANDOFF triggered! Reason: {reason}")
                print(f"  📍 Position: ({x:.1f}, {y:.1f}, {z:.1f})")
                print(f"  🌡️ Smoke density: {self.last_smoke_density * 100:.2f}%")
                print(f"  📡 Altitude: {abs(self.current_altitude):.1f}m AGL")

                # Hover for smooth transition
                self.client.hoverAsync().join()

                self.transition_to_ppo = True
                return {
                    "reason": reason,
                    "position": self.get_position(),
                    "smoke_density": self.last_smoke_density,
                    "fire_visible": self.fire_detected,
                    "altitude": self.current_altitude,
                    "elapsed": elapsed,
                    "step": step
                }

            time.sleep(STEP_DURATION * 0.5)


def main():
    """Standalone test for plume tracking."""
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("🛫 Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    # Climb to starting altitude
    client.moveToZAsync(START_ALTITUDE, 5).join()
    time.sleep(1)

    tracker = PlumeTracker(client, START_ALTITUDE)
    result = tracker.track_plume()

    print(f"\n📊 Result: {result}")
    print(f"📝 Flight log: {len(tracker.flight_log)} entries")

    # Land after test
    print("🔽 Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Plume tracking test complete!")


if __name__ == "__main__":
    main()
