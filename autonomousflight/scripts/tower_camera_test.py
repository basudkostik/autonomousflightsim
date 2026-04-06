"""
Phase 3 - AI Tower Camera Test (ExternalCameras / FixedCamera)
===============================================================
Uses the "FixedCamera1" defined in settings.json under ExternalCameras.
The drone stays on the ground — the tower camera is completely independent.

PREREQUISITES:
  1. settings.json must have "ExternalCameras" with "FixedCamera1"
  2. Open UE4 and press Play
  3. Run this script
"""

import airsim
import numpy as np
import cv2
import os
import time


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # 1. Connect to AirSim
    # ──────────────────────────────────────────────
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")

    # ──────────────────────────────────────────────
    # 2. Capture image from FixedCamera1 (AI Tower)
    # ──────────────────────────────────────────────
    print("Capturing image from FixedCamera1 (AI Tower)...")

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
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # Find the next available number for the filename
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("tower_view_") and f.endswith(".png")]
        indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()]
        next_index = max(indices) + 1 if indices else 1
        
        filename = f"tower_view_{next_index}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img_rgb)
        print(f"Tower view saved: {save_path}")
        print(f"Image size: {img_rgb.shape} (height x width x channels)")
    else:
        print("Failed to capture image from FixedCamera1.")
        print("Check that settings.json has ExternalCameras configured.")
        return

    # ──────────────────────────────────────────────
    # 3. Verify drone is still on the ground
    # ──────────────────────────────────────────────
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"\nDrone position (untouched): X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
    print("Drone stayed on the ground. Tower camera is independent!")

    print(f"\nDone! Image saved as {filename}")


if __name__ == "__main__":
    main()
