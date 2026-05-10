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
import sys

# (ai_tower_monitor imports removed — fire position is now passed in directly)

# Flush stdout immediately so DummyVecEnv wrapper doesn't buffer our prints
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
IMAGE_SIZE = 84                     # Observation image resolution
MAX_STEPS_PER_EPISODE = 500         # Max steps per episode
STEP_DURATION = 1.0                 # 1 Hz control loop (was 0.5)

# Navigation bounds (AirSim meters)
NAV_ALTITUDE = -5                   # ~5m above ground (under canopy)
MAX_ALTITUDE = -2                   # Don't go too low
MIN_ALTITUDE = -10                  # Don't go too high in PPO phase

# Homing phase — direct fly-toward-fire when far away
HOMING_DISTANCE = 20.0             # Metres: beyond this, use direct homing not PPO
HOMING_SPEED = 5.0                 # m/s direct homing speed (was 10, too fast)

# Descent phase — start descending when within this distance
DESCENT_NEAR_FIRE = 20.0           # Metres: start descending when closer than this
DESCENT_STEP = 0.5                 # Metres to descend per step
DESCENT_FLOOR_Z = 0.0             # Floor Z: origin altitude (user says we can go under 0)

# 360° scan phase — take 4 photos when very close to fire
SCAN_DISTANCE = 5.0                # Metres: trigger 360° scan when closer than this

# Action limits (PPO phase only — homing phase uses HOMING_SPEED)
MAX_FORWARD_VEL = 12.0              # m/s (increased for faster approach)
MAX_LATERAL_VEL = 2.0               # m/s
MAX_YAW_RATE = 45.0                 # degrees/s

# Reward weights
REWARD_PROGRESS = 2.0               # Progress signal toward fire (lowered with slower speed)
REWARD_SURVIVAL = 0.1
REWARD_COLLISION = -10.0
REWARD_FIRE_CONFIRMED = 50.0
REWARD_TIME_PENALTY = -0.05

# Fire confirmation distance (meters)
FIRE_CONFIRM_DISTANCE = 5.0

# Fire detection HSV (widened to catch yellows + oranges from above)
FIRE_HSV_LOWER = np.array([0, 50, 150])
FIRE_HSV_UPPER = np.array([40, 255, 255])

# Distance threshold for automatic confirmation (no HSV needed)
# If the drone is this close, fire is confirmed by proximity alone
FIRE_PROXIMITY_CONFIRM = 2.0

# Spawn area configuration
SPAWN_RADIUS_MIN = 20.0            # Min distance from fire to spawn
SPAWN_RADIUS_MAX = 50.0            # Max distance from fire to spawn


class AirSimFireEnv(gym.Env):
    """
    Custom Gymnasium environment for under-canopy fire finding with PPO.

    Fire position and tower bearing are provided at construction time—  
    no map generation or tower monitoring happens inside the environment.
    Each episode simply resets the drone near the known fire location.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, fire_x=0.0, fire_y=0.0, tower_bearing=0.0, render_mode=None, is_chained=False, nav_altitude=None):
        """
        Args:
            fire_x, fire_y: Known fire position in AirSim meters (e.g. from fire_placer.py
                            or from the PlumeTracker handoff).
            tower_bearing: The bearing angle the AI Tower detected (degrees).
                           Used by the PPO agent as navigation hint.
            render_mode:   "human" for visualization.
            is_chained:    True when created by PlumeTracker's chain — drone is already
                           airborne and API-controlled, so skip hard resets.
        """
        super().__init__()

        self.render_mode = render_mode
        self.is_chained = is_chained
        # Use passed altitude if given (e.g. flying over mountain), else default
        self.nav_altitude = nav_altitude if nav_altitude is not None else NAV_ALTITUDE

        # Fire position — set at construction, stays fixed for all episodes
        self.fire_x = fire_x
        self.fire_y = fire_y

        # Bearing from the AI Tower — used as the PPO target-direction hint
        self.tower_bearing = tower_bearing

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
        # Forward vel range is [0, MAX] — drone always moves forward (never backwards).
        # This ensures even random/untrained policies produce net forward motion.
        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_LATERAL_VEL, -MAX_YAW_RATE], dtype=np.float32),
            high=np.array([MAX_FORWARD_VEL, MAX_LATERAL_VEL, MAX_YAW_RATE], dtype=np.float32),
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
        Calculate direction FROM the drone's current position TOWARD the fire.
        Returns a unit vector in the drone's local frame.

        Previous version used tower_bearing (a fixed compass heading) which
        gives the PPO no useful signal at large distances. The fire position
        is already known (fire_x, fire_y), so we use it directly.
        """
        x, y, z = self._get_position()
        dx = self.fire_x - x
        dy = self.fire_y - y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.1:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Normalize to unit vector (world frame)
        dx /= dist
        dy /= dist

        # Convert world-frame direction to drone-local frame
        yaw = math.radians(self._get_orientation())
        local_dx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_dy = dx * math.sin(-yaw) + dy * math.cos(-yaw)

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
          1. If distance < FIRE_PROXIMITY_CONFIRM → confirmed by proximity alone
             (fire particles viewed from above don't always match HSV range)
          2. If distance < FIRE_CONFIRM_DISTANCE → check for fire-colored pixels
        """
        dist = self._distance_to_fire()

        # Proximity-only confirmation — drone is practically on top of the fire
        if dist <= FIRE_PROXIMITY_CONFIRM:
            return True

        if dist > FIRE_CONFIRM_DISTANCE:
            return False

        rgb = self._get_rgb_image()
        if rgb is None:
            return False

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
        fire_pixels = cv2.countNonZero(fire_mask)

        return fire_pixels > 50

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
        Reset environment for a new training episode.

        Fire position and tower bearing are already known (set at construction).
        This reset ONLY handles drone repositioning — no map generation,
        no tower detection, no Phase A or Phase B.

        Uses a SOFT RESET instead of client.reset() to avoid destroying
        the AirSim API control state mid-chain. client.reset() disarms
        the drone and disables API control, which causes "API call not
        received" errors and safety mode on subsequent commands.
        """
        super().reset(seed=seed)
        self._connect()
        self.episode_count += 1
        self.step_count = 0

        print(f"\n" + "-" * 50)
        print(f"  🔄 Episode {self.episode_count} — resetting drone near fire...", flush=True)
        print(f"  🔥 Fire: ({self.fire_x:.1f}, {self.fire_y:.1f}) m | Bearing hint: {self.tower_bearing:.1f}°", flush=True)
        print("-" * 50)

        # ── Soft Reset ───────────────────────────────────────────
        # NEVER call client.reset() here — it kills API control and
        # causes "API call not received" → safety mode → crash loop.
        # Instead: hover → re-arm → reposition.
        try:
            self.client.hoverAsync().join()
        except Exception:
            pass
        time.sleep(0.3)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.3)

        # Check if drone is already airborne (chained from PlumeTracker)
        _, _, current_z = self._get_position()
        already_airborne = current_z < -1.0  # More than 1m above ground

        if not already_airborne:
            print("  🛫 Taking off...", flush=True)
            self.client.takeoffAsync().join()
            time.sleep(0.5)

        # Move to PPO nav altitude (inherits mountain altitude if chained)
        print(f"  📡 Moving to nav altitude {abs(self.nav_altitude):.1f}m...", flush=True)
        self.client.moveToZAsync(self.nav_altitude, 5).join()
        time.sleep(0.5)

        # Face toward fire so first observation is meaningful
        x, y, z = self._get_position()
        dx = self.fire_x - x
        dy = self.fire_y - y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist > 0.5:
            yaw = math.degrees(math.atan2(dy, dx))
            self.client.rotateToYawAsync(yaw).join()
        time.sleep(0.3)

        self.prev_distance = self._distance_to_fire()
        obs = self._get_observation()
        info = {
            "episode": self.episode_count,
            "spawn_distance": self.prev_distance,
            "fire_position": (self.fire_x, self.fire_y)
        }

        print(f"  ✅ Reset done. Distance to fire: {self.prev_distance:.1f}m", flush=True)
        return obs, info


    def _do_360_scan(self):
        """
        Take 4 photos at 90° intervals with camera at -15° pitch.
        Provides 360° coverage for fire confirmation from directly above.
        Returns True if fire detected in any of the 4 photos.
        """
        print(f"\n  📸 360° SCAN — fire within {SCAN_DISTANCE}m!", flush=True)
        pitch_rad = math.radians(-45.0)  # Look steeply downward to see fire below
        fire_seen = False

        for i, yaw_offset in enumerate([0, 90, 180, 270]):
            # Rotate drone to each quadrant
            x, y, z = self._get_position()
            dx = self.fire_x - x
            dy = self.fire_y - y
            base_yaw = math.degrees(math.atan2(dy, dx))
            target_yaw = base_yaw + yaw_offset

            try:
                self.client.rotateToYawAsync(target_yaw).join()
                time.sleep(0.3)
            except Exception:
                pass

            # Set camera to -15° pitch for this scan direction
            camera_pose = airsim.Pose(
                airsim.Vector3r(0, 0, 0),
                airsim.to_quaternion(pitch_rad, 0, 0)
            )
            self.client.simSetCameraPose("front_center", camera_pose)
            time.sleep(0.1)

            # Capture and save the scan photo
            rgb = self._get_rgb_image()
            if rgb is not None:
                fname = os.path.join(
                    self.output_dir,
                    f"scan360_{yaw_offset}deg_ep{self.episode_count}.png"
                )
                cv2.imwrite(fname, rgb)
                print(f"    📷 Scan {i+1}/4 ({yaw_offset}°) saved → {os.path.basename(fname)}", flush=True)

                # Check for fire in this frame
                hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                fire_mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
                if cv2.countNonZero(fire_mask) > 30:
                    fire_seen = True
                    print(f"    🔥 Fire detected in scan {yaw_offset}°!", flush=True)

        return fire_seen

    def step(self, action):
        """
        Execute one step of the environment.

        Three behavioural phases based on distance to fire:
          dist > HOMING_DISTANCE : Direct homing (fly straight toward fire at HOMING_SPEED)
          dist < DESCENT_NEAR_FIRE: Begin descending 0.5m/step
          dist < SCAN_DISTANCE   : Perform 360° scan + proximity confirm
        """
        self.step_count += 1
        current_distance = self._distance_to_fire()
        x, y, z = self._get_position()

        # ─── Phase 1: DIRECT HOMING (dist > 20m) ──────────────
        # PPO is untrained at start — using its random actions would cause
        # the drone to wander. Instead, fly directly toward fire until close.
        if current_distance > HOMING_DISTANCE:
            dx = self.fire_x - x
            dy = self.fire_y - y
            dist = math.sqrt(dx**2 + dy**2)
            vx = HOMING_SPEED * dx / dist
            vy = HOMING_SPEED * dy / dist
            # Also yaw toward fire for meaningful observations
            target_yaw = math.degrees(math.atan2(dy, dx))
            self.client.rotateToYawAsync(target_yaw)
            self.client.moveByVelocityZAsync(
                vx, vy, self.nav_altitude,
                duration=STEP_DURATION,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw)
            )

        # ─── Phase 2: PPO CONTROL (dist ≤ 20m) ────────────────
        else:
            forward_vel = float(np.clip(action[0], 0, MAX_FORWARD_VEL))
            lateral_vel = float(np.clip(action[1], -MAX_LATERAL_VEL, MAX_LATERAL_VEL))
            yaw_rate = float(np.clip(action[2], -MAX_YAW_RATE, MAX_YAW_RATE))
            yaw = math.radians(self._get_orientation())
            vx = forward_vel * math.cos(yaw) - lateral_vel * math.sin(yaw)
            vy = forward_vel * math.sin(yaw) + lateral_vel * math.cos(yaw)

            # Descend when close to fire
            target_alt = self.nav_altitude
            if current_distance < DESCENT_NEAR_FIRE:
                # AirSim NED: more positive Z = lower altitude (toward ground)
                # z + DESCENT_STEP goes DOWN (was z - DESCENT_STEP which went UP!)
                new_alt = z + DESCENT_STEP
                target_alt = min(new_alt, DESCENT_FLOOR_Z)  # Don't go below floor

            self.client.moveByVelocityZAsync(
                vx, vy, target_alt,
                duration=STEP_DURATION,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            )

        time.sleep(STEP_DURATION)

        # ─── 360° Scan when very close ─────────────────────────
        scan_confirmed = False
        if current_distance < SCAN_DISTANCE:
            scan_confirmed = self._do_360_scan()

        # ─── Calculate Reward ───
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self._check_collision():
            reward += REWARD_COLLISION
            terminated = True
            info["termination"] = "collision"

        # Re-measure distance after move
        current_distance = self._distance_to_fire()
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            reward += REWARD_PROGRESS * progress
        self.prev_distance = current_distance

        reward += REWARD_SURVIVAL
        reward += REWARD_TIME_PENALTY

        # Fire confirmation: proximity, scan, or HSV
        if not terminated and (scan_confirmed or self._check_fire_confirmed()):
            reward += REWARD_FIRE_CONFIRMED
            terminated = True
            info["termination"] = "fire_confirmed"
            info["steps_to_confirm"] = self.step_count
            print(f"\n  🔥 FIRE CONFIRMED at {current_distance:.1f}m!", flush=True)

        if self.step_count >= MAX_STEPS_PER_EPISODE:
            truncated = True
            info["termination"] = "max_steps"

        obs = self._get_observation()
        x, y, z = self._get_position()
        info.update({
            "step": self.step_count,
            "distance_to_fire": current_distance,
            "reward": reward
        })

        mode = "HOMING" if current_distance > HOMING_DISTANCE else "PPO"
        print(f"\r  Phase C | Step {self.step_count:3d} | "
              f"Dist: {current_distance:6.1f}m | "
              f"Alt: {abs(z):4.1f}m | "
              f"Mode: {mode:6s} | "
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
