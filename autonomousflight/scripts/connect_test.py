"""
Phase 2 - Step 1: Basic AirSim Connection Test
================================================
This script connects Python to the Unreal Engine simulation via AirSim.
It performs a simple takeoff → fly forward → hover → land sequence.

PREREQUISITES:
  1. Unreal Engine must be running with AirSim plugin
  2. Click Play (▶) in the UE4 editor before running this script
  3. Install dependencies: pip install -r requirements.txt
"""

import airsim
import time


def main():
    # ──────────────────────────────────────────────
    # 1. Connect to AirSim (UE4 must be playing!)
    # ──────────────────────────────────────────────
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected to AirSim!")

    # ──────────────────────────────────────────────
    # 2. Enable API control (Python takes over)
    # ──────────────────────────────────────────────
    client.enableApiControl(True)
    print("✅ API control enabled")

    # ──────────────────────────────────────────────
    # 3. Arm the drone motors
    # ──────────────────────────────────────────────
    client.armDisarm(True)
    print("✅ Drone armed")

    # ──────────────────────────────────────────────
    # 4. Take off
    # ──────────────────────────────────────────────
    print("🚁 Taking off...")
    client.takeoffAsync().join()
    print("✅ Airborne!")

    # ──────────────────────────────────────────────
    # 5. Fly forward at 5 m/s for 3 seconds
    # ──────────────────────────────────────────────
    print("➡️  Flying forward...")
    client.moveByVelocityAsync(5, 0, 0, 3).join()

    # ──────────────────────────────────────────────
    # 6. Hover in place for 2 seconds
    # ──────────────────────────────────────────────
    client.hoverAsync().join()
    print("✅ Hovering")
    time.sleep(2)

    # ──────────────────────────────────────────────
    # 7. Land the drone
    # ──────────────────────────────────────────────
    print("🔽 Landing...")
    client.landAsync().join()

    # ──────────────────────────────────────────────
    # 8. Disarm and release API control
    # ──────────────────────────────────────────────
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Done! Drone landed safely.")


if __name__ == "__main__":
    main()
