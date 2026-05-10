"""
Phase A — Bearing Navigator
============================
Bearing-based flight toward fire with automatic descent.

FLOW:
  1. Receive bearing angle from AI Tower
  2. Convert bearing to AirSim velocity vector
  3. Fly at cruise altitude toward the bearing
  4. Continuously capture RGB from drone camera
  5. After DESCENT_START_TIME seconds, begin gradual descent
  6. When smoke detected OR timeout → chain directly to Phase C (PPO Training)

PlumeTracker is no longer used — BearingNavigator handles both
cruise and descent phases, then hands off to PPO.

USAGE:
  Called by ai_tower_monitor dispatch_drone, or standalone:
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
# AirSim NED: z is positive downward, so altitude AGL is negative (e.g. -5 ≈ 5 m up).
DEFAULT_CRUISE_ALTITUDE = -5
DEFAULT_SPEED = 6.0                 # m/s cruise speed
SMOKE_DENSITY_THRESHOLD = 0.15     # 15% of image pixels showing smoke → transition
SMOKE_CHECK_INTERVAL = 1.0         # Seconds between smoke density checks
MAX_FLIGHT_TIME = 1200             # Safety: max seconds before handing off to PPO
VELOCITY_STEP_DURATION = 2.0       # Duration of each velocity command
MIN_STEPS_BEFORE_SMOKE_CHECK = 10  # Must fly at least this many steps before allowing
                                    # smoke handoff — prevents instant handoff at spawn

# Camera pitch angle (degrees, negative = look down)
# Without this, the camera looks straight ahead at sky/mountains,
# which triggers false 65% "smoke" from bright sky pixels.
CAMERA_PITCH_DEG = -30.0           # Tilt camera 30° downward
CAMERA_YAW_DEG = -20.0             # Left offset so fire (left side) is seen earlier

# ── Descent Phase Configuration ──────────────────────────
# After flying for DESCENT_START_TIME seconds at cruise altitude,
# the drone starts descending gradually toward the fire.
DESCENT_START_TIME = 240           # Seconds before descent begins
DESCENT_RATE = 0.3                 # Meters per step to descend (positive = descend)
MIN_DESCENT_ALTITUDE = -3          # Lowest altitude before PPO handoff (3m AGL)

# HSV range for smoke/fire detection
# Narrowed to avoid false positives from:
#   - River (high V, low S, blue hue ~100-130) → excluded by H < 30 or H > 170
#   - Bright sky (V very high but S=0) → excluded by requiring some saturation
#   - Green terrain (H~60) → excluded by hue range
# Target: grey/white smoke (low S, high V, any hue BUT not blue)
SMOKE_HSV_LOWER1 = np.array([0,   0,  180])   # White/grey smoke (hue 0-30)
SMOKE_HSV_UPPER1 = np.array([30,  60, 255])   # ..up to slightly warm grey
SMOKE_HSV_LOWER2 = np.array([150, 0,  180])   # Hue wrap-around (350-360°)
SMOKE_HSV_UPPER2 = np.array([180, 60, 255])   # Pure grey/white from red side
# River exclusion: blue water (H 90-130, high V, low S)
RIVER_HSV_LOWER = np.array([90,  0,  150])
RIVER_HSV_UPPER = np.array([130, 80, 255])

# Fire keywords to search for in scene object names
FIRE_KEYWORDS = ["fire", "flame", "bp_fire"]


def detect_fire_position(client):
    """
    Auto-detect the fire blueprint position from the active AirSim scene.
    Uses the same approach as map_generator.py: fetch ALL scene objects
    with no regex filter, then filter by name in Python.
    (simListSceneObjects with regex like '*Fire*' hangs because * is invalid regex)

    Prints fire location to console.
    Returns (fire_x, fire_y) or (0.0, 0.0) if not found.
    """
    try:
        all_objects = client.simListSceneObjects()
        fire_actors = [
            obj for obj in all_objects
            if any(kw in obj.lower() for kw in FIRE_KEYWORDS)
        ]

        if fire_actors:
            pose = client.simGetObjectPose(fire_actors[0])
            x = pose.position.x_val
            y = pose.position.y_val
            print(f"  🔥 Fire actor detected : {fire_actors[0]}")
            print(f"  📍 Fire AirSim position: ({x:.2f}, {y:.2f}) m")
            if len(fire_actors) > 1:
                print(f"  ℹ️  Multiple fire actors found ({len(fire_actors)}), using first")
            return x, y
    except Exception as e:
        print(f"  ⚠️ Error detecting fire: {e}")

    print("  ⚠️  No fire actor found in scene.")
    return 0.0, 0.0


class BearingNavigator:
    """
    Phase A: Fly toward a bearing angle, descend after a set time,
    then chain directly to Phase C (PPO Training).
    PlumeTracker is no longer used.
    """

    def __init__(self, client, bearing, altitude=DEFAULT_CRUISE_ALTITUDE,
                 speed=DEFAULT_SPEED, smoke_threshold=SMOKE_DENSITY_THRESHOLD,
                 fire_x=None, fire_y=None):
        """
        Args:
            client: airsim.MultirotorClient (already connected)
            bearing: Target bearing angle in degrees (0=North, 90=East)
            altitude: AirSim Z coordinate (negative = up)
            speed: Cruise speed in m/s
            smoke_threshold: Fraction of pixels that must be smoke to trigger transition
            fire_x, fire_y: Known fire position (auto-detected before takeoff)
        """
        self.client = client
        self.bearing = bearing
        self.altitude = altitude
        self.current_altitude = altitude  # Tracks altitude during descent
        self.speed = speed
        self.smoke_threshold = smoke_threshold

        self.start_time = None
        self.transition_triggered = False
        self.last_smoke_density = 0.0
        self.flight_log = []
        self.descending = False  # True once descent phase begins

        # Fire position — passed in from dispatch_drone (detected before takeoff)
        self.fire_x = fire_x if fire_x is not None else 0.0
        self.fire_y = fire_y if fire_y is not None else 0.0

        # Output directory for debug images
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "bearing_nav")
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_cte_velocity(self, dx, dy):
        """
        Calculate velocity vector using Cross-Track Error (CTE) to stay on the Tower's ray.
        This forces the drone to "beam ride" the exact mathematical line extending from the Tower
        regardless of where the Drone started flying from.
        """
        # Ray direction vector
        bearing_rad = math.radians(self.bearing)
        rx = math.cos(bearing_rad)
        ry = math.sin(bearing_rad)
        
        # Tower coordinates (assumed 0,0 where AI Tower sits)
        tx, ty = 0.0, 0.0
        
        # Vector from Tower to Drone
        vec_x = dx - tx
        vec_y = dy - ty
        
        # Cross-Track Error (2D Cross Product between ray and drone vector)
        # Positive if Drone is 'left' of the ray, negative if 'right'
        cte = rx * vec_y - ry * vec_x
        
        # Proportional gain for steering back to the ray mathematically
        k_cte = 0.5 
        
        # Corrective velocity (perpendicular to ray)
        corr_vx = ry * cte * k_cte
        corr_vy = -rx * cte * k_cte
        
        # Combine forward cruise velocity with lateral CTE correction
        vx = (self.speed * rx) + corr_vx
        vy = (self.speed * ry) + corr_vy
        
        return vx, vy, cte

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
        Uses two hue ranges to catch warm-grey smoke while excluding:
          - Blue river (H 90-130)
          - Pure sky (handled by requiring V < 255 or moderate S)

        Returns:
            float: Smoke density ratio (0.0 to 1.0)
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Two-range smoke mask (avoids blue river region in the middle)
        mask1 = cv2.inRange(hsv, SMOKE_HSV_LOWER1, SMOKE_HSV_UPPER1)
        mask2 = cv2.inRange(hsv, SMOKE_HSV_LOWER2, SMOKE_HSV_UPPER2)
        smoke_mask = cv2.bitwise_or(mask1, mask2)

        # Subtract river pixels so blue water doesn't count as smoke
        river_mask = cv2.inRange(hsv, RIVER_HSV_LOWER, RIVER_HSV_UPPER)
        smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(river_mask))

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

        total_pixels = smoke_mask.shape[0] * smoke_mask.shape[1]
        smoke_pixels = cv2.countNonZero(smoke_mask)
        density = smoke_pixels / total_pixels

        return density

    def should_transition(self):
        """Check if we should transition to PPO training."""
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
            "smoke_density": smoke_density,
            "descending": self.descending
        }
        self.flight_log.append(entry)
        return entry

    def _chain_to_ppo_training(self):
        """
        Chain directly to Phase C: train a PPO model on-site.

        PlumeTracker is skipped entirely — BearingNavigator handles
        both cruise and descent, then hands off to PPO training.
        """
        from ppo_environment import AirSimFireEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

        # Stabilize drone and HOLD ALTITUDE during PPO setup
        # The model creation (importing torch, building neural net) takes
        # several seconds — without active flight commands the drone freefalls.
        print("\n  🔄 Stabilizing drone before PPO handoff...", flush=True)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        try:
            self.client.hoverAsync().join()
        except Exception:
            pass

        # Hold altitude at current position while we set up the model
        x, y, z = self.get_position()
        hold_alt = z if z < -1.0 else -5.0  # Use current altitude or default 5m
        print(f"  📡 Holding altitude at Z={hold_alt:.1f}m during setup...", flush=True)
        self.client.moveToZAsync(hold_alt, 2)  # Non-blocking, keeps drone in place
        time.sleep(0.5)

        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        # Use detected fire position (from scene) rather than drone position
        fire_x = self.fire_x if self.fire_x != 0.0 or self.fire_y != 0.0 else x
        fire_y = self.fire_y if self.fire_x != 0.0 or self.fire_y != 0.0 else y

        print("\n" + "=" * 55)
        print("  🧠 PHASE C — PPO ON-SITE TRAINING")
        print("=" * 55)
        print(f"  📍 Drone position  : ({x:.1f}, {y:.1f}) m")
        print(f"  🔥 Fire position   : ({fire_x:.1f}, {fire_y:.1f}) m")
        print(f"  🎯 Tower bearing   : {self.bearing:.1f}°")
        print(f"  📁 Model output    : {model_dir}/ppo_forest_final.zip")
        print("-" * 55)

        # Re-issue altitude hold — model imports below take several seconds
        self.client.moveToZAsync(hold_alt, 2)

        # Create environment — fire position already known, no tower/map-gen
        # is_chained=True tells the env the drone is already airborne
        # nav_altitude=hold_alt keeps PPO at mountain altitude (not hardcoded -5)
        def make_env():
            env = AirSimFireEnv(
                fire_x=fire_x,
                fire_y=fire_y,
                tower_bearing=self.bearing,
                is_chained=True,
                nav_altitude=hold_alt
            )
            return Monitor(env)

        env = DummyVecEnv([make_env])

        # Re-issue altitude hold — PPO() constructor takes a few seconds
        self.client.moveToZAsync(hold_alt, 2)

        # Build PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
        )

        # Callback to STOP training once fire is confirmed
        # Without this, PPO resets → confirms fire instantly → resets → infinite loop
        class FireConfirmedCallback(BaseCallback):
            def _on_step(self):
                infos = self.locals.get("infos", [])
                for info in infos:
                    if info.get("termination") == "fire_confirmed":
                        print("\n\n  ✅ Fire confirmed — stopping PPO training!", flush=True)
                        return False  # Stops model.learn()
                return True

        print("\n  🚀 Training started — will stop when fire is confirmed")
        total_steps = 50000
        try:
            model.learn(
                total_timesteps=total_steps,
                callback=[checkpoint, FireConfirmedCallback()],
                progress_bar=True
            )
            print(f"\n  ✅ Training complete")
        except KeyboardInterrupt:
            print("\n  ⛔ Training interrupted — saving current weights...")

        model_path = os.path.join(model_dir, "ppo_forest_final")
        model.save(model_path)
        print(f"  💾 Model saved → {model_path}.zip")

        env.close()

        return {
            "fire_confirmed": True,
            "reason": "ppo_training_complete",
            "position": (x, y, z),
            "model_path": f"{model_path}.zip"
        }

    def fly_toward_bearing(self):
        """
        Main flight loop: fly toward bearing at cruise altitude.
        After DESCENT_START_TIME seconds, begin gradual descent.
        Hands off to PPO training when smoke detected, altitude low enough, or timeout.

        Returns:
            dict: Final state with position and reason for transition
        """
        print("\n" + "=" * 55)
        print("  🧭 PHASE A — BEARING NAVIGATION")
        print("=" * 55)
        print(f"  Target bearing     : {self.bearing:.1f}°")
        print(f"  Cruise altitude    : {abs(self.altitude)}m AGL")
        print(f"  Cruise speed       : {self.speed} m/s")
        print(f"  Smoke threshold    : {self.smoke_threshold * 100:.1f}%")
        print(f"  Max flight time    : {MAX_FLIGHT_TIME}s")
        print(f"  Descent starts at  : {DESCENT_START_TIME}s")
        print(f"  Descent rate       : {DESCENT_RATE} m/step")
        print(f"  Min descent alt    : {abs(MIN_DESCENT_ALTITUDE)}m AGL")
        if self.fire_x != 0.0 or self.fire_y != 0.0:
            print(f"  🔥 Fire location   : ({self.fire_x:.1f}, {self.fire_y:.1f}) m")
        print("-" * 55)

        # Tilt the drone camera downward + slightly left so it sees
        # the ground/smoke instead of sky (which triggers false smoke density)
        pitch_rad = math.radians(CAMERA_PITCH_DEG)
        yaw_rad = math.radians(CAMERA_YAW_DEG)
        camera_pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(pitch_rad, 0, yaw_rad)  # pitch, roll, yaw
        )
        self.client.simSetCameraPose("front_center", camera_pose)
        print(f"  📷 Camera: pitch {CAMERA_PITCH_DEG}° down, yaw {CAMERA_YAW_DEG}° left")

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

        self.start_time = time.time()
        step = 0

        while True:
            step += 1
            elapsed = time.time() - self.start_time

            # Safety timeout — hand off to PPO training directly
            if elapsed > MAX_FLIGHT_TIME:
                print(f"\n  ⏱️ Timeout after {MAX_FLIGHT_TIME}s — handing off to PPO training")
                self.client.hoverAsync().join()
                return self._chain_to_ppo_training()

            # ── Descent Phase ─────────────────────────────────────
            # After DESCENT_START_TIME, begin gradual descent
            if elapsed >= DESCENT_START_TIME:
                if not self.descending:
                    self.descending = True
                    print(f"\n\n  📉 DESCENT PHASE started at {elapsed:.0f}s — lowering altitude gradually")

                # Descend by DESCENT_RATE per step (altitude becomes less negative → lower)
                self.current_altitude = min(
                    self.current_altitude + DESCENT_RATE,
                    MIN_DESCENT_ALTITUDE
                )
                # Command descent
                self.client.moveToZAsync(self.current_altitude, 2)

            # Continuously calculate Cross-Track velocity to beam-ride the AI Tower's ray
            dx, dy, dz = self.get_position()
            vx, vy, cte = self.compute_cte_velocity(dx, dy)

            # Cap total velocity magnitude to cruise speed to avoid safety limits
            total_speed = math.sqrt(vx ** 2 + vy ** 2)
            if total_speed > self.speed:
                vx = vx / total_speed * self.speed
                vy = vy / total_speed * self.speed

            # FIX: Use MaxDegreeOfFreedom instead of ForwardOnly.
            # ForwardOnly auto-rotates drone to face the velocity vector.
            # When combined with yaw_mode (fixed bearing), these fight each other
            # and cause AirSim to enter safety/failsafe mode.
            # MaxDegreeOfFreedom lets us control yaw explicitly and independently.
            self.client.moveByVelocityAsync(
                vx, vy, 0,
                duration=VELOCITY_STEP_DURATION,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
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

                descent_str = " 📉" if self.descending else "   "
                print(f"\r  Step {step:4d} | "
                      f"Pos: ({x:8.1f}, {y:8.1f}) | "
                      f"Alt: {abs(z):5.1f}m | "
                      f"CTE: {cte:5.1f}m | "
                      f"Smoke: {self.last_smoke_density * 100:5.2f}% | "
                      f"Time: {elapsed:5.1f}s{descent_str}",
                      end="", flush=True)

                # Save debug image every 10 steps
                if step % 10 == 0:
                    debug_path = os.path.join(self.output_dir, f"bearing_step_{step}.png")
                    cv2.imwrite(debug_path, rgb)

                # Check transition → chain directly to PPO training
                # Guard: must have flown MIN_STEPS_BEFORE_SMOKE_CHECK steps first
                # Prevents instant handoff when fire is right at spawn
                if step >= MIN_STEPS_BEFORE_SMOKE_CHECK and self.should_transition():
                    print(f"\n\n  🔥 SMOKE DETECTED! Density: {self.last_smoke_density * 100:.2f}%")
                    print(f"  ✅ Handing off to Phase C — PPO Training")

                    # Save transition frame
                    cv2.imwrite(
                        os.path.join(self.output_dir, f"transition_frame_{step}.png"),
                        rgb
                    )

                    self.client.hoverAsync().join()
                    self.transition_triggered = True
                    return self._chain_to_ppo_training()

                # Also transition when descent reaches minimum altitude
                if self.descending and self.current_altitude >= MIN_DESCENT_ALTITUDE:
                    print(f"\n\n  📉 Minimum altitude {abs(MIN_DESCENT_ALTITUDE)}m reached!")
                    print(f"  ✅ Handing off to Phase C — PPO Training")

                    self.client.hoverAsync().join()
                    self.transition_triggered = True
                    return self._chain_to_ppo_training()


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
