"""
Phase C — PPO Navigator (Inference Mode)
==========================================
Loads a trained PPO model and navigates at low altitude to find fire.

Used by MissionController after Phase B hands off control.

USAGE:
  Called by MissionController, or standalone:
    python scripts/ppo_navigator.py --model models/ppo_forest_final.zip

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Trained PPO model (.zip file)
  3. Drone already airborne at low altitude
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

from stable_baselines3 import PPO


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
IMAGE_SIZE = 84
NAV_ALTITUDE = -5                   # ~5m AGL (under canopy)
STEP_DURATION = 0.1                 # 10 Hz
MAX_STEPS = 300                     # Max steps before aborting
FIRE_CONFIRM_DISTANCE = 5.0        # Meters

# Action limits (must match training env)
MAX_FORWARD_VEL = 4.0
MAX_LATERAL_VEL = 2.0
MAX_YAW_RATE = 45.0

# Fire detection HSV
FIRE_HSV_LOWER = np.array([0, 100, 200])
FIRE_HSV_UPPER = np.array([25, 255, 255])


class PPONavigator:
    """
    Phase C: PPO-based navigation at low altitude for fire finding.
    Uses a pre-trained model for inference.
    """

    def __init__(self, client, model_path, fire_estimate=None):
        """
        Args:
            client: airsim.MultirotorClient (already connected)
            model_path: Path to trained PPO model (.zip)
            fire_estimate: (x, y) estimated fire position from plume tracker
        """
        self.client = client
        self.fire_estimate = fire_estimate or (50.0, 50.0)

        # Load trained model
        print(f"  🧠 Loading PPO model: {model_path}")
        self.model = PPO.load(model_path)
        print(f"  ✅ Model loaded!")

        self.step_count = 0
        self.fire_confirmed = False
        self.flight_log = []

        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "ppo_nav")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_position(self):
        """Get drone position."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def get_orientation(self):
        """Get drone yaw in degrees."""
        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        yaw = math.atan2(
            2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
            1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
        )
        return math.degrees(yaw)

    def get_depth_image(self):
        """Capture and process depth image for the model."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True)
            ])
            if responses and responses[0].height > 0:
                depth = airsim.list_to_2d_float_array(
                    responses[0].image_data_float,
                    responses[0].width,
                    responses[0].height
                )
                depth = np.clip(depth, 0, 100) / 100.0
                depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE))
                return depth.reshape(IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32)
        except Exception:
            pass
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)

    def get_target_direction(self):
        """Calculate direction to estimated fire in drone-local frame."""
        x, y, z = self.get_position()
        yaw = math.radians(self.get_orientation())

        dx = self.fire_estimate[0] - x
        dy = self.fire_estimate[1] - y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.1:
            return np.array([0.0, 0.0], dtype=np.float32)

        dx /= dist
        dy /= dist

        local_dx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_dy = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        return np.array([local_dx, local_dy], dtype=np.float32)

    def get_observation(self):
        """Build observation dict matching training env."""
        return {
            "depth": self.get_depth_image(),
            "target_direction": self.get_target_direction()
        }

    def check_fire_confirmed(self):
        """Check if fire is visually confirmed."""
        x, y, z = self.get_position()
        dist = math.sqrt(
            (self.fire_estimate[0] - x) ** 2 +
            (self.fire_estimate[1] - y) ** 2
        )

        if dist > FIRE_CONFIRM_DISTANCE:
            return False

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

                hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                fire_mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
                fire_pixels = cv2.countNonZero(fire_mask)

                if fire_pixels > 100:
                    # Save confirmation image
                    cv2.imwrite(
                        os.path.join(self.output_dir, "fire_confirmed.png"),
                        rgb
                    )
                    return True
        except Exception:
            pass

        return False

    def check_collision(self):
        """Check for collision."""
        collision = self.client.simGetCollisionInfo()
        return collision.has_collided

    def navigate_and_confirm(self):
        """
        Main navigation loop using trained PPO model.

        Returns:
            dict: Result with fire_confirmed status and final position
        """
        print("\n" + "=" * 55)
        print("  🧠 PHASE C — PPO NAVIGATION (Under Canopy)")
        print("=" * 55)
        print(f"  🔥 Fire estimate: ({self.fire_estimate[0]:.1f}, {self.fire_estimate[1]:.1f})")
        print(f"  📡 Altitude: {abs(NAV_ALTITUDE)}m AGL")
        print(f"  ⏱️ Max steps: {MAX_STEPS}")
        print("-" * 55)

        # Ensure correct altitude
        self.client.moveToZAsync(NAV_ALTITUDE, 3).join()
        time.sleep(0.5)

        start_time = time.time()

        for step in range(1, MAX_STEPS + 1):
            self.step_count = step

            # Get observation
            obs = self.get_observation()

            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)

            # Unpack and clip action
            forward_vel = float(np.clip(action[0], -MAX_FORWARD_VEL, MAX_FORWARD_VEL))
            lateral_vel = float(np.clip(action[1], -MAX_LATERAL_VEL, MAX_LATERAL_VEL))
            yaw_rate = float(np.clip(action[2], -MAX_YAW_RATE, MAX_YAW_RATE))

            # Convert to world frame
            yaw_rad = math.radians(self.get_orientation())
            vx = forward_vel * math.cos(yaw_rad) - lateral_vel * math.sin(yaw_rad)
            vy = forward_vel * math.sin(yaw_rad) + lateral_vel * math.cos(yaw_rad)

            # Execute
            self.client.moveByVelocityAsync(
                vx, vy, 0,
                duration=STEP_DURATION,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            )
            self.client.moveToZAsync(NAV_ALTITUDE, 2)
            time.sleep(STEP_DURATION)

            # Check state
            x, y, z = self.get_position()
            dist = math.sqrt(
                (self.fire_estimate[0] - x) ** 2 +
                (self.fire_estimate[1] - y) ** 2
            )

            # Log
            self.flight_log.append({
                "step": step, "x": x, "y": y, "z": z,
                "distance": dist,
                "action": [forward_vel, lateral_vel, yaw_rate]
            })

            print(f"\r  Step {step:3d}/{MAX_STEPS} | "
                  f"Pos: ({x:7.1f}, {y:7.1f}) | "
                  f"Dist: {dist:6.1f}m | "
                  f"Act: ({forward_vel:+.1f}, {lateral_vel:+.1f}, {yaw_rate:+.0f}°/s)",
                  end="", flush=True)

            # Check collision
            if self.check_collision():
                print(f"\n\n  💥 COLLISION detected! Aborting.")
                self.client.hoverAsync().join()
                return {
                    "fire_confirmed": False,
                    "reason": "collision",
                    "position": (x, y, z),
                    "steps": step,
                    "elapsed": time.time() - start_time
                }

            # Check fire confirmation
            if self.check_fire_confirmed():
                self.fire_confirmed = True
                print(f"\n\n  🔥✅ FIRE CONFIRMED!")
                print(f"  📍 Position: ({x:.1f}, {y:.1f}, {z:.1f})")
                print(f"  📏 Distance: {dist:.1f}m")
                print(f"  ⏱️ Steps: {step}")

                self.client.hoverAsync().join()
                return {
                    "fire_confirmed": True,
                    "reason": "fire_confirmed",
                    "position": (x, y, z),
                    "distance": dist,
                    "steps": step,
                    "elapsed": time.time() - start_time
                }

        # Max steps reached
        print(f"\n\n  ⏱️ Max steps reached without fire confirmation.")
        self.client.hoverAsync().join()
        return {
            "fire_confirmed": False,
            "reason": "max_steps",
            "position": self.get_position(),
            "steps": MAX_STEPS,
            "elapsed": time.time() - start_time
        }


def main():
    """Standalone test for PPO navigation."""
    parser = argparse.ArgumentParser(description="Phase C — PPO Navigator")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "models", "ppo_forest_final.zip"),
                        help="Path to trained PPO model")
    parser.add_argument("--fire-x", type=float, default=50.0,
                        help="Estimated fire X position")
    parser.add_argument("--fire-y", type=float, default=50.0,
                        help="Estimated fire Y position")
    args = parser.parse_args()

    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("🛫 Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    # Move to low altitude
    client.moveToZAsync(NAV_ALTITUDE, 5).join()
    time.sleep(1)

    nav = PPONavigator(client, args.model, (args.fire_x, args.fire_y))
    result = nav.navigate_and_confirm()

    print(f"\n📊 Result: {result}")
    print(f"📝 Flight log: {len(nav.flight_log)} entries")

    # Land
    print("🔽 Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ PPO navigation test complete!")


if __name__ == "__main__":
    main()
