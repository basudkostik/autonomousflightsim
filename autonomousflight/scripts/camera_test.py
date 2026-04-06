"""
Phase 2 - Step 2: Camera Image Capture Test
=============================================
This script connects to AirSim, takes off, captures RGB and Depth
images from the drone's front camera, saves them as PNG files, and lands.

PREREQUISITES:
  1. Unreal Engine must be running with AirSim plugin
  2. Click Play (▶) in the UE4 editor before running this script
  3. Install dependencies: pip install -r requirements.txt
"""

import airsim
import numpy as np
import cv2
import os
import time


def main():
    # Output directory for saved images
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # 1. Connect to AirSim
    # ──────────────────────────────────────────────
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!")

    client.enableApiControl(True)
    client.armDisarm(True)

    # ──────────────────────────────────────────────
    # 2. Take off and move up a bit
    # ──────────────────────────────────────────────
    print("🚁 Taking off...")
    client.takeoffAsync().join()

    # Fly up a bit to get a better view (z is negative = up in AirSim)
    client.moveToZAsync(-5, 2).join()
    time.sleep(1)

    # ──────────────────────────────────────────────
    # 3. Capture RGB Image
    # ──────────────────────────────────────────────
    print("📷 Capturing RGB image...")
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    ])

    rgb_response = responses[0]
    img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(rgb_response.height, rgb_response.width, 3)

    rgb_path = os.path.join(output_dir, "rgb_image.png")
    cv2.imwrite(rgb_path, img_rgb)
    print(f"✅ RGB image saved: {rgb_path}")
    print(f"   Image size: {img_rgb.shape} (height x width x channels)")

    # ──────────────────────────────────────────────
    # 4. Capture Depth Image
    # ──────────────────────────────────────────────
    print("📷 Capturing Depth image...")
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True),
    ])

    depth_response = responses[0]
    depth_img = airsim.list_to_2d_float_array(
        depth_response.image_data_float,
        depth_response.width,
        depth_response.height
    )

    # Normalize depth to 0-255 for visualization
    depth_normalized = np.clip(depth_img, 0, 100)  # Clip to 100m max
    depth_visual = np.array(depth_normalized / 100.0 * 255, dtype=np.uint8)

    depth_path = os.path.join(output_dir, "depth_image.png")
    cv2.imwrite(depth_path, depth_visual)
    print(f"✅ Depth image saved: {depth_path}")
    print(f"   Image size: {depth_visual.shape} (height x width)")
    print(f"   Depth range: {depth_img.min():.1f}m — {depth_img.max():.1f}m")

    # ──────────────────────────────────────────────
    # 5. Get drone state info
    # ──────────────────────────────────────────────
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"\n📍 Drone Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

    # ──────────────────────────────────────────────
    # 6. Land and cleanup
    # ──────────────────────────────────────────────
    print("\n🔽 Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Done! Check the 'output' folder for saved images.")


if __name__ == "__main__":
    main()
