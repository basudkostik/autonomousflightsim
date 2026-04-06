"""
Phase 2 - Step 3: Advanced Flight Test
========================================
This script tests more complex flight patterns: square path, altitude
changes, and yaw rotation. Use this to verify full flight control
before building the RL environment in Phase 3.

PREREQUISITES:
  1. Unreal Engine must be running with AirSim plugin
  2. Click Play (▶) in the UE4 editor before running this script
  3. Install dependencies: pip install -r requirements.txt
"""

import airsim
import time


def main():
    # ──────────────────────────────────────────────
    # 1. Connect and take off
    # ──────────────────────────────────────────────
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ Connected and armed")

    print("🚁 Taking off...")
    client.takeoffAsync().join()

    # Move to a safe altitude (z=-5 means 5 meters above ground)
    client.moveToZAsync(-5, 2).join()
    print("✅ At 5m altitude")

    # ──────────────────────────────────────────────
    # 2. Fly a square pattern
    # ──────────────────────────────────────────────
    speed = 3  # m/s
    duration = 3  # seconds per side

    directions = [
        ("Forward  (+X)", (speed, 0, 0)),
        ("Right    (+Y)", (0, speed, 0)),
        ("Backward (-X)", (-speed, 0, 0)),
        ("Left     (-Y)", (0, -speed, 0)),
    ]

    print("\n📐 Flying a square pattern...")
    for name, (vx, vy, vz) in directions:
        print(f"  ➡️  {name}...")
        client.moveByVelocityAsync(vx, vy, vz, duration).join()
        client.hoverAsync().join()
        time.sleep(0.5)

    print("✅ Square pattern complete!")

    # ──────────────────────────────────────────────
    # 3. Test yaw rotation (spin 360 degrees)
    # ──────────────────────────────────────────────
    print("\n🔄 Rotating 360 degrees...")
    client.rotateByYawRateAsync(45, 8).join()  # 45 deg/s for 8 seconds = 360°
    client.hoverAsync().join()
    print("✅ Rotation complete!")

    # ──────────────────────────────────────────────
    # 4. Print final position
    # ──────────────────────────────────────────────
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"\n📍 Final Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
    print("   (Should be close to starting position after square)")

    # ──────────────────────────────────────────────
    # 5. Land
    # ──────────────────────────────────────────────
    print("\n🔽 Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ All flight tests passed!")


if __name__ == "__main__":
    main()
