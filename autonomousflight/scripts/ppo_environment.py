"""
Phase C — PPO Training Environment
====================================
Custom Gymnasium environment wrapping AirSim for PPO training.

The agent navigates at low altitude under the forest canopy,
avoids tree collisions, and finds + confirms fire.

OBSERVATION SPACE:
  - Depth image: 84×84 grayscale (normalized 0-1)
  - Target direction: 2D vector (dx, dy) pointing toward estimated fire

ACTION SPACE:
  - Continuous: [forward_velocity, lateral_velocity, yaw_rate]

REWARD:
  +1.0   progress toward fire
  +0.1   survival per step
  -10.0  collision (episode ends)
  +50.0  fire confirmed
  -0.05  each step (time penalty to encourage efficiency)

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Forest scene with fire actor placed
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2
import time
import math
import os


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
IMAGE_SIZE = 84                     # Observation image resolution
MAX_STEPS_PER_EPISODE = 300         # ~30 seconds at 10 Hz
STEP_DURATION = 0.1                 # 10 Hz control loop

# Navigation bounds (AirSim meters)
NAV_ALTITUDE = -5                   # ~5m above ground (under canopy)
MAX_ALTITUDE = -2                   # Don't go too low
MIN_ALTITUDE = -10                  # Don't go too high in PPO phase

# Action limits
MAX_FORWARD_VEL = 4.0               # m/s
MAX_LATERAL_VEL = 2.0               # m/s
MAX_YAW_RATE = 45.0                 # degrees/s

# Reward weights
REWARD_PROGRESS = 1.0
REWARD_SURVIVAL = 0.1
REWARD_COLLISION = -10.0
REWARD_FIRE_CONFIRMED = 50.0
REWARD_TIME_PENALTY = -0.05

# Fire confirmation distance (meters)
FIRE_CONFIRM_DISTANCE = 5.0

# Fire detection HSV
FIRE_HSV_LOWER = np.array([0, 100, 200])
FIRE_HSV_UPPER = np.array([25, 255, 255])

# Spawn area configuration
SPAWN_RADIUS_MIN = 20.0            # Min distance from fire to spawn
SPAWN_RADIUS_MAX = 50.0            # Max distance from fire to spawn


class AirSimFireEnv(gym.Env):
    """
    Custom Gymnasium environment for under-canopy fire finding with PPO.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: "human" for visualization
        """
        super().__init__()

        self.render_mode = render_mode

        # Fire position will be set by map generator
        self.fire_x, self.fire_y = 0.0, 0.0

        # ─── Observation Space ───
        # Dict space: depth image + target direction
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                low=0.0, high=1.0,
                shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                dtype=np.float32
            ),
            "target_direction": spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        })

        # ─── Action Space ───
        # [forward_vel, lateral_vel, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-MAX_FORWARD_VEL, -MAX_LATERAL_VEL, -MAX_YAW_RATE]),
            high=np.array([MAX_FORWARD_VEL, MAX_LATERAL_VEL, MAX_YAW_RATE]),
            dtype=np.float32
        )

        # AirSim client
        self.client = None
        self.connected = False
        self.step_count = 0
        self.prev_distance = None
        self.episode_count = 0

        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "ppo_training")
        os.makedirs(self.output_dir, exist_ok=True)

    def _connect(self):
        """Establish AirSim connection if not connected."""
        if not self.connected:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.connected = True

    def _get_position(self):
        """Get drone position as (x, y, z)."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def _get_orientation(self):
        """Get drone yaw in degrees."""
        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        # Convert quaternion to yaw
        yaw = math.atan2(
            2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
            1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
        )
        return math.degrees(yaw)

    def _distance_to_fire(self):
        """Calculate horizontal distance to fire."""
        x, y, z = self._get_position()
        dx = self.fire_x - x
        dy = self.fire_y - y
        return math.sqrt(dx ** 2 + dy ** 2)

    def _get_target_direction(self):
        """
        Calculate normalized direction vector from drone to fire
        in the drone's local frame.

        Returns:
            np.array: [dx, dy] normalized direction vector
        """
        x, y, z = self._get_position()
        yaw = math.radians(self._get_orientation())

        # World-frame direction to fire
        world_dx = self.fire_x - x
        world_dy = self.fire_y - y
        dist = math.sqrt(world_dx ** 2 + world_dy ** 2)

        if dist < 0.1:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Normalize
        world_dx /= dist
        world_dy /= dist

        # Rotate into drone-local frame
        local_dx = world_dx * math.cos(-yaw) - world_dy * math.sin(-yaw)
        local_dy = world_dx * math.sin(-yaw) + world_dy * math.cos(-yaw)

        return np.array([local_dx, local_dy], dtype=np.float32)

    def _get_depth_image(self):
        """
        Capture depth image and resize to IMAGE_SIZE × IMAGE_SIZE.

        Returns:
            np.array: Normalized depth image (0-1), shape (84, 84, 1)
        """
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
                # Clip and normalize (0-100m range → 0-1)
                depth = np.clip(depth, 0, 100) / 100.0
                # Resize
                depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE))
                return depth.reshape(IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32)
        except Exception:
            pass

        # Fallback: empty depth image
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)

    def _get_rgb_image(self):
        """Capture RGB image for fire detection."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                return img1d.reshape(responses[0].height, responses[0].width, 3)
        except Exception:
            pass
        return None

    def _check_collision(self):
        """Check if drone has collided."""
        collision = self.client.simGetCollisionInfo()
        return collision.has_collided

    def _check_fire_confirmed(self):
        """
        Check if fire is confirmed:
          1. Distance to fire < threshold
          2. Fire-colored pixels visible in RGB camera
        """
        dist = self._distance_to_fire()
        if dist > FIRE_CONFIRM_DISTANCE:
            return False

        rgb = self._get_rgb_image()
        if rgb is None:
            return False

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
        fire_pixels = cv2.countNonZero(fire_mask)

        return fire_pixels > 100

    def _get_observation(self):
        """Build the observation dict."""
        depth = self._get_depth_image()
        target_dir = self._get_target_direction()
        return {
            "depth": depth,
            "target_direction": target_dir
        }

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.
        Executes full procedural pipeline:
        1. Generate new map (trees + fire)
        2. Tower calculates bearing to fire
        3. Phase A: Bearing Nav
        4. Phase B: Plume Tracking
        5. Phase C: PPO start (returns observation)
        """
        super().reset(seed=seed)
        self._connect()
        self.episode_count += 1
        self.step_count = 0

        print(f"\n" + "=" * 50)
        print(f"🎬 STARTING EPISODE {self.episode_count} — FULL PIPELINE (Map -> A -> B -> C)")
        print(f"=" * 50)

        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 1. MAP GENERATION
        # Pass seed=None to get a random map every episode
        from map_generator import MapGenerator
        map_gen = MapGenerator(self.client, seed=None)
        map_gen.generate()
        
        self.fire_x, self.fire_y = map_gen.get_fire_position()

        # 2. AI TOWER — same idea as ai_tower_monitor: compare consecutive tower frames
        # so we react to NEW smoke motion, not a single-frame bright blob (sky/clouds).
        print(f"\n🗼 AI TOWER: Monitoring forest for smoke (frame differencing)...")
        from ai_tower_monitor import (
            capture_tower_image,
            compare_frames,
            calculate_bearing as tower_calculate_bearing,
            CAMERA_FOV,
            TOWER_YAW,
            CHANGE_THRESHOLD,
            CHANGE_AREA_MIN,
        )

        # Map generates fire randomly. Instantly take the first picture.
        print("   📸 Capturing first baseline photo...")
        prev_frame = capture_tower_image(self.client)

        # Wait 7 seconds for the smoke particle system to spawn and rise above the trees
        print("   ⏳ Waiting 7 seconds for smoke to rise above tree line...")
        time.sleep(7)

        # Take second photo
        print("   📸 Capturing second photo for comparison...")
        curr_frame = capture_tower_image(self.client)

        detected_bearing = None
        if prev_frame is not None and curr_frame is not None:
            _, _, change_area, change_center = compare_frames(
                prev_frame, curr_frame, CHANGE_THRESHOLD
            )

            if change_area >= CHANGE_AREA_MIN and change_center is not None:
                cx, cy = change_center
                img_w = curr_frame.shape[1]
                detected_bearing = tower_calculate_bearing(cx, img_w, CAMERA_FOV, TOWER_YAW)
                print(f"   🚨🔥 CHANGE DETECTED! Area ~{change_area:.0f} px at ({cx}, {cy})")
                print(f"   🎯 Bearing from motion: {detected_bearing:.1f}°")

        # Fallback to mathematical if visual detection fails (to keep training moving)
        if detected_bearing is None:
            print("   ⚠️ Visual detection failed or timed out (smoke hidden behind trees).")
            drone_start_x, drone_start_y = 0.0, 0.0
            dx = self.fire_x - drone_start_x
            dy = self.fire_y - drone_start_y
            detected_bearing = math.degrees(math.atan2(dy, dx))
            if detected_bearing < 0:
                detected_bearing += 360.0
            print(f"   📐 Using exact mathematical bearing instead: {detected_bearing:.1f}°")

        # Takeoff ONLY after tower dispatches
        print("\n🚁 TOWER SIGNAL RECEIVED -> TAKING OFF...")
        
        # TAKEOFF SEQUENCE (Modeled exactly after the successful a.py test)
        print("   🚁 Executing native Takeoff...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(1)
        
        # Now command the absolute altitude using the tested moveToZAsync
        print("   🛫 Climbing to safe cruise altitude Z: -20.0m...")
        self.client.moveToZAsync(-20.0, 5) # Async background call! No join()
        
        # Monitor progress live so we can see if it's stuck!
        climb_start = time.time()
        while time.time() - climb_start < 15:
            current_z = self.client.getMultirotorState().kinematics_estimated.position.z_val
            print(f"   [Debug Z] Climbing... Current Z: {current_z:.2f}m")
            if current_z <= -18.0: # Negative is UP! Close enough
                print("   ✅ Reached initial climb altitude!")
                break
            time.sleep(1)

        # 3. PHASE A: BEARING NAVIGATION
        from bearing_navigator import BearingNavigator
        nav = BearingNavigator(self.client, detected_bearing, speed=8.0) 
        nav.fly_toward_bearing()
        
        # 4. PHASE B: PLUME TRACKING
        # Faster descent for training
        from plume_tracker import PlumeTracker
        tracker = PlumeTracker(self.client, descent_rate=1.0, speed=4.0)
        tracker.track_plume()

        # PHASE C Setup:
        # Move drone to exactly NAV_ALTITUDE to start PPO
        self.client.moveToZAsync(NAV_ALTITUDE, 5).join()
        
        # Face toward the exact fire right at handoff to give PPO sensible start
        x, y, z = self._get_position()
        dx = self.fire_x - x
        dy = self.fire_y - y
        yaw = math.degrees(math.atan2(dy, dx))
        self.client.rotateToYawAsync(yaw).join()
        time.sleep(0.5)

        self.prev_distance = self._distance_to_fire()

        obs = self._get_observation()
        info = {
            "episode": self.episode_count,
            "spawn_distance": self.prev_distance,
            "fire_position": (self.fire_x, self.fire_y)
        }

        return obs, info

    def step(self, action):
        """
        Execute one step of the environment.

        Args:
            action: np.array [forward_vel, lateral_vel, yaw_rate]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Unpack action
        forward_vel = float(np.clip(action[0], -MAX_FORWARD_VEL, MAX_FORWARD_VEL))
        lateral_vel = float(np.clip(action[1], -MAX_LATERAL_VEL, MAX_LATERAL_VEL))
        yaw_rate = float(np.clip(action[2], -MAX_YAW_RATE, MAX_YAW_RATE))

        # Convert to world-frame velocity
        yaw = math.radians(self._get_orientation())
        vx = forward_vel * math.cos(yaw) - lateral_vel * math.sin(yaw)
        vy = forward_vel * math.sin(yaw) + lateral_vel * math.cos(yaw)

        # Execute action
        self.client.moveByVelocityAsync(
            vx, vy, 0,
            duration=STEP_DURATION,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
        # Maintain altitude
        self.client.moveToZAsync(NAV_ALTITUDE, 2)
        time.sleep(STEP_DURATION)

        # ─── Calculate Reward ───
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Check collision
        if self._check_collision():
            reward += REWARD_COLLISION
            terminated = True
            info["termination"] = "collision"

        # Progress reward
        current_distance = self._distance_to_fire()
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            reward += REWARD_PROGRESS * progress
        self.prev_distance = current_distance

        # Survival reward
        reward += REWARD_SURVIVAL

        # Time penalty
        reward += REWARD_TIME_PENALTY

        # Fire confirmation
        if not terminated and self._check_fire_confirmed():
            reward += REWARD_FIRE_CONFIRMED
            terminated = True
            info["termination"] = "fire_confirmed"
            info["steps_to_confirm"] = self.step_count

        # Truncation (max steps)
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            truncated = True
            info["termination"] = "max_steps"

        # Build observation
        obs = self._get_observation()

        info.update({
            "step": self.step_count,
            "distance_to_fire": current_distance,
            "reward": reward
        })

        # Progress logging
        x, y, z = self._get_position()
        print(f"\r  Phase C | Step {self.step_count:3d} | "
              f"Dist: {current_distance:6.1f}m | "
              f"Alt: {abs(z):4.1f}m | "
              f"Reward: {reward:+.2f}", 
              end="", flush=True)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up."""
        if self.connected and self.client:
            try:
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except Exception:
                pass

    def render(self):
        """Optional rendering — saves debug images."""
        if self.render_mode == "human":
            rgb = self._get_rgb_image()
            if rgb is not None:
                cv2.imwrite(
                    os.path.join(self.output_dir,
                                 f"ep{self.episode_count}_step{self.step_count}.png"),
                    rgb
                )
