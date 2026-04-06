"""
Phase 3 - AI Tower Smoke Detection & Bearing Calculation
=========================================================
This script processes the tower view image to detect smoke/fire
plumes and calculates the bearing angle toward the target.

FUNCTIONS:
  1. Load the latest tower_view image
  2. Apply HSV color masking for smoke (gray/white)
  3. Find the largest contour (smoke plume)
  4. Calculate its pixel center
  5. Convert pixel position to Bearing Angle based on Camera FOV
"""

import cv2
import numpy as np
import os
import math

def calculate_bearing(pixel_x, image_width, camera_fov, tower_yaw=0):
    """
    Calculates the bearing angle toward a pixel.
    pixel_x: The X coordinate of the smoke in the image
    image_width: Total width of the image
    camera_fov: Horizontal field of view of the camera
    tower_yaw: The direction the tower is facing in the world
    """
    # Calculate offset from center (-0.5 to 0.5)
    relative_offset = (pixel_x / image_width) - 0.5
    
    # Calculate angle offset from center
    angle_offset = relative_offset * camera_fov
    
    # Final bearing in world coordinates
    bearing = (tower_yaw + angle_offset) % 360
    return bearing

def detect_smoke(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Image not found"
    
    h, w = img.shape[:2]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color range for smoke (Gray/White/Light Blue plumes)
    # Note: These values might need tuning depending on your forest lighting!
    lower_smoke = np.array([0, 0, 150])    # Low saturation, high value (white/gray)
    upper_smoke = np.array([180, 50, 255]) 
    
    # Create mask
    mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
    
    # Apply some noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, "No smoke detected in the frame"
    
    # Filter by area to ignore tiny noise
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not valid_contours:
        return None, "No significant smoke plumes found"
    
    # Get the largest plume
    largest_cnt = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(largest_cnt)
    
    if M["m00"] == 0:
        return None, "Calculation error"
        
    # Calculate center of mass
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Draw results on a copy for verification
    result_img = img.copy()
    cv2.drawContours(result_img, [largest_cnt], -1, (0, 255, 0), 2)
    cv2.circle(result_img, (cx, cy), 7, (255, 0, 0), -1)
    
    return (cx, cy, w, h, result_img), "Success"

def main():
    # Paths
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    
    # Find the latest tower_view photo
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("tower_view_") and f.endswith(".png")]
    if not existing_files:
        print("❌ No tower_view images found in output folder. Run tower_camera_test.py first.")
        return
        
    # Sort to get the latest (highest index)
    indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()]
    latest_index = max(indices)
    image_name = f"tower_view_{latest_index}.png"
    image_path = os.path.join(output_dir, image_name)
    
    print(f"🔍 Analyzing {image_name} for smoke...")
    
    result, status = detect_smoke(image_path)
    
    if result:
        cx, cy, w, h, result_img = result
        print("✅ Detection Successful!")
        print(f"   Smoke Center: Pixel ({cx}, {cy})")
        
        # Calculate bearing
        # Settings from our JSON: FOV = 150, Yaw = 0
        tower_yaw = 0 
        camera_fov = 150
        
        bearing = calculate_bearing(cx, w, camera_fov, tower_yaw)
        
        print(f"🎯 CALCULATED BEARING: {bearing:.2f} degrees")
        
        # Save verification image
        verif_path = os.path.join(output_dir, f"detected_smoke_{latest_index}.png")
        cv2.imwrite(verif_path, result_img)
        print(f"📝 Verification image saved to: {verif_path}")
        
    else:
        print(f"❌ Detection Failed: {status}")

if __name__ == "__main__":
    main()
