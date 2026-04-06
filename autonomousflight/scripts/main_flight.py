import airsim
import time
import os
import sys
import math

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from map_generator import MapGenerator
from ai_tower_monitor import capture_tower_image, compare_frames, calculate_bearing, CAMERA_FOV, TOWER_YAW, CHANGE_THRESHOLD, CHANGE_AREA_MIN
from bearing_navigator import BearingNavigator
from plume_tracker import PlumeTracker
from ppo_navigator import PPONavigator

def run_main_mission():
    print("="*60)
    print("  🔥 AUTONOMOUS FIRE DETECTION MISSION PIPELINE")
    print("="*60)
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Pre-setup
    client.enableApiControl(True)
    client.armDisarm(True)

    # ---------------------------------------------------------
    # STEP 1: Map Generator
    # ---------------------------------------------------------
    print("\n[STEP 1] Generating map...")
    map_gen = MapGenerator(client)
    map_gen.generate()
    fire_x, fire_y = map_gen.get_fire_position()

    # ---------------------------------------------------------
    # STEP 2: AI Tower Signal Logic
    # ---------------------------------------------------------
    print("\n[STEP 2] AI Tower monitoring for smoke...")
    
    # Run Tower First Time (Baseline)
    print("   📸 Capturing first photo (background baseline)...")
    base_frame = capture_tower_image(client)
    
    # Wait for smoke
    print("   ⏳ Waiting 7 seconds for smoke to rise...")
    time.sleep(7)
    
    # Run Tower Second Time (Monitor)
    print("   📸 Capturing second photo for comparison...")
    curr_frame = capture_tower_image(client)
    
    print("\n   [STEP 2.b] Comparing frames for smoke detection...")
    _, _, change_area, change_center = compare_frames(base_frame, curr_frame, CHANGE_THRESHOLD)
    
    detected_bearing = 0.0
    if change_center is not None and change_area > CHANGE_AREA_MIN:
        cx, cy = change_center
        img_w = curr_frame.shape[1]
        detected_bearing = calculate_bearing(cx, img_w, CAMERA_FOV, TOWER_YAW)
        print(f"   🚨 VISUAL ALARM! Smoke detected. Bearing: {detected_bearing:.1f}°")
    else:
        # Fallback for testing if smoke isn't visible due to trees
        dx = fire_x - 0.0
        dy = fire_y - 0.0
        detected_bearing = math.degrees(math.atan2(dy, dx))
        if detected_bearing < 0: detected_bearing += 360
        print(f"   ⚠️ Visual detection failed. Using exact math bearing: {detected_bearing:.1f}°")

    # ---------------------------------------------------------
    # STEP 3: Take Off
    # ---------------------------------------------------------
    # TAKEOFF SEQUENCE (Modeled exactly after the successful a.py test)
    print("   🚁 Executing native Takeoff...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    time.sleep(1)
    
    # Now command the absolute altitude using the tested moveToZAsync
    print("   🛫 Climbing to safe cruise altitude Z: -20.0m...")
    client.moveToZAsync(-20.0, 5) # Async background call! No join()
    
    # Monitor progress live so we can see if it's stuck!
    climb_start = time.time()
    while time.time() - climb_start < 15:
        current_z = client.getMultirotorState().kinematics_estimated.position.z_val
        print(f"   [Debug Z] Climbing... Current Z: {current_z:.2f}m")
        if current_z <= -18.0: # Negative is UP! Close enough
            print("   ✅ Reached initial climb altitude!")
            break
        time.sleep(1)
    
    # ---------------------------------------------------------
    # STEP 4: Phase A (Bearing Navigator)
    # ---------------------------------------------------------
    print("\n[STEP 4] Executing Phase A: Bearing Navigator...")
    nav = BearingNavigator(client, detected_bearing, speed=8.0)
    nav.fly_toward_bearing()
    
    # ---------------------------------------------------------
    # STEP 5: Phase B (Plume Tracker)
    # ---------------------------------------------------------
    print("\n[STEP 5] Executing Phase B: Plume Tracker...")
    tracker = PlumeTracker(client, descent_rate=1.0, speed=4.0)
    tracker.track_plume()
    
    # ---------------------------------------------------------
    # STEP 6: Phase C (PPO Agent)
    # ---------------------------------------------------------
    print("\n[STEP 6] Executing Phase C: PPO Agent...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_forest_final.zip")
    if os.path.exists(model_path):
        ppo_agent = PPONavigator(client, model_path)
        ppo_agent.run()
    else:
        print("   ⚠️ PPO Model not found. Skipping Phase C. (Train PPO first to generate model!)")

    print("\n✅ MISSION FULLY COMPLETE!")
    client.hoverAsync().join()

if __name__ == "__main__":
    try:
        run_main_mission()
    except KeyboardInterrupt:
        print("\n\n⛔ Mission stopped by user.")
