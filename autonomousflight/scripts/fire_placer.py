"""
Fire Placer
===========
Randomly places one fire actor within a configured UE4 bounding box.

HANDLES MOUNTAIN TERRAIN:
  Because terrain is not flat, we cannot simply set Z=0. Instead:

  1. At startup, we read the fire actor's CURRENT Z from the AirSim scene.
     (In UE4 editor the designer placed it at the correct terrain height.)
     This gives us at least one valid ground-level Z reference.

  2. You can add more terrain samples to TERRAIN_HEIGHT_SAMPLES_UE4 below.
     Measure them in UE4 editor: click ground → Details panel → Location Z.
     The more samples you add, the more accurate Z will be across the mountain.

  3. For any random (X, Y), we estimate the terrain Z via nearest-neighbor
     lookup from the sample list.

BOUNDING BOX (UE4 cm):
  (41500, 1500) → (41500, -22500) → (20000, -22500) → (20000, 1500)

USAGE:
  python scripts/fire_placer.py
  python scripts/fire_placer.py --seed 42
  python scripts/fire_placer.py --list      (just list fire actors)
"""

import airsim
import math
import random
import argparse
import sys
import os

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Bounding box for fire placement (UE4 world coordinates in cm)
# Corners: (41500, 1500) / (41500, -22500) / (20000, -22500) / (20000, 1500)
BOUNDS_UE4_X = (20000, 41500)    # (min_x, max_x) — UE4 cm
BOUNDS_UE4_Y = (-22500, 1500)    # (min_y, max_y) — UE4 cm

# Drone / PlayerStart position in UE4 world coordinates (cm)
# AirSim maps this exact point to (0, 0, 0) in NED meters.
DRONE_START_UE4_CM = (20800, -2700, 7700)

# Fire actor name search patterns (tried in order)
FIRE_ACTOR_PATTERNS = ["*Fire*", "*fire*", "*Flame*", "*BP_Fire*"]

# ── Mountain Terrain Height Samples ──────────────────────
# Add terrain Z values you measured in UE4 editor below.
# In UE4: click on the ground at a location → Details → Location Z = terrain Z
#
# Format: (ue4_x_cm, ue4_y_cm, ue4_z_cm)
#
# The more points you add, the more accurately fire is placed on slopes.
# If this list is EMPTY, the fire actor's current editor Z is used for all positions.
TERRAIN_HEIGHT_SAMPLES_UE4 = [
    # Example (fill in your actual values from UE4 editor):
    # (25000, -5000,  9200),
    # (30000, -10000, 11500),
    # (38000, -18000, 8800),
]

# ═══════════════════════════════════════════════════════


def ue4_to_airsim(x_ue4, y_ue4, z_ue4):
    """Convert UE4 world cm coordinates to AirSim NED meters."""
    x = (x_ue4 - DRONE_START_UE4_CM[0]) / 100.0
    y = (y_ue4 - DRONE_START_UE4_CM[1]) / 100.0
    z = -(z_ue4 - DRONE_START_UE4_CM[2]) / 100.0  # Negate: UE4 Z-up → NED Z-down
    return x, y, z


def airsim_to_ue4(x, y, z):
    """Convert AirSim NED meters to UE4 world cm coordinates."""
    x_ue4 = int(x * 100 + DRONE_START_UE4_CM[0])
    y_ue4 = int(y * 100 + DRONE_START_UE4_CM[1])
    z_ue4 = int(-z * 100 + DRONE_START_UE4_CM[2])
    return x_ue4, y_ue4, z_ue4


def find_fire_actors(client):
    """Find all fire-related actor names in the active AirSim scene."""
    found = set()
    for pattern in FIRE_ACTOR_PATTERNS:
        try:
            objects = client.simListSceneObjects(name_regex=pattern)
            found.update(objects)
        except Exception:
            pass
    return sorted(found)


def estimate_terrain_z_ue4(target_x, target_y, fallback_z):
    """
    Estimate terrain Z (UE4 cm) at (target_x, target_y) via nearest-neighbor
    lookup in TERRAIN_HEIGHT_SAMPLES_UE4.

    If no samples are configured, returns fallback_z (from the fire actor's
    current editor-placed position).
    """
    samples = list(TERRAIN_HEIGHT_SAMPLES_UE4)
    if not samples:
        return fallback_z

    best_dist = float('inf')
    best_z = fallback_z
    for (sx, sy, sz) in samples:
        dist = math.sqrt((target_x - sx) ** 2 + (target_y - sy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_z = sz

    return best_z


def place_fire(client, seed=None):
    """
    Randomly place a fire actor within the configured bounding box.

    Args:
        client: Connected airsim.MultirotorClient
        seed:   Optional int seed for repeatable placement

    Returns:
        dict with placement info, or None on failure.
    """
    print("\n" + "=" * 55)
    print("  🔥 FIRE PLACER")
    print("=" * 55)

    if seed is not None:
        random.seed(seed)
        print(f"  Seed: {seed}")

    # ── Find fire actors ────────────────────────────────
    fire_actors = find_fire_actors(client)
    if not fire_actors:
        print("  ❌ No fire actors found in scene!")
        print("     Make sure your UE4 level has an actor with 'Fire' in its name.")
        print("     Patterns searched:", FIRE_ACTOR_PATTERNS)
        return None

    fire_name = fire_actors[0]
    print(f"  Fire actors found : {fire_actors}")
    print(f"  Using actor       : {fire_name}")

    # ── Get fire actor's current pose (editor-placed terrain Z) ────
    try:
        current_pose = client.simGetObjectPose(fire_name)
    except Exception as e:
        print(f"  ❌ Cannot read fire actor pose: {e}")
        return None

    cur_ue4_x, cur_ue4_y, cur_ue4_z = airsim_to_ue4(
        current_pose.position.x_val,
        current_pose.position.y_val,
        current_pose.position.z_val
    )
    print(f"  Current fire UE4  : ({cur_ue4_x}, {cur_ue4_y}, {cur_ue4_z}) cm")
    print(f"  Current fire AirSim: ({current_pose.position.x_val:.2f}, "
          f"{current_pose.position.y_val:.2f}, {current_pose.position.z_val:.2f}) m")

    # ── Pick random (X, Y) within bounds ───────────────
    rand_x = random.uniform(*BOUNDS_UE4_X)
    rand_y = random.uniform(*BOUNDS_UE4_Y)

    # ── Estimate terrain Z at the random position ───────
    # Uses nearest-neighbor from TERRAIN_HEIGHT_SAMPLES_UE4.
    # Falls back to the fire actor's current editor Z if no samples are defined.
    terrain_z_ue4 = estimate_terrain_z_ue4(rand_x, rand_y, cur_ue4_z)

    z_source = "nearest terrain sample" if TERRAIN_HEIGHT_SAMPLES_UE4 else "fire actor editor Z"
    print(f"\n  📍 Random target  : UE4 ({rand_x:.0f}, {rand_y:.0f}) cm")
    print(f"  🏔️  Terrain Z      : {terrain_z_ue4} cm  ({z_source})")

    # ── Convert to AirSim and build pose ───────────────
    ax, ay, az = ue4_to_airsim(rand_x, rand_y, terrain_z_ue4)

    new_pose = airsim.Pose(
        position_val=airsim.Vector3r(float(ax), float(ay), float(az)),
        orientation_val=current_pose.orientation   # Keep editor rotation
    )

    print(f"  AirSim target     : ({ax:.2f}, {ay:.2f}, {az:.2f}) m")

    # ── Place fire ─────────────────────────────────────
    try:
        success = client.simSetObjectPose(fire_name, new_pose, teleport=True)
    except Exception as e:
        print(f"  ❌ simSetObjectPose failed: {e}")
        return None

    if success:
        print(f"\n  ✅ Fire placed successfully at UE4 ({rand_x:.0f}, {rand_y:.0f}, {terrain_z_ue4:.0f})")
    else:
        print(f"  ⚠️  simSetObjectPose returned False — actor may not have moved.")
        print(f"     Common cause: 'Movable' not enabled on the actor in UE4 editor.")
        print(f"     Fix: Select actor → Details → Transform → Mobility = Movable")

    # ── Mountain terrain reminder ────────────────────────
    if not TERRAIN_HEIGHT_SAMPLES_UE4:
        print("\n  ⚠️  MOUNTAIN TERRAIN: No TERRAIN_HEIGHT_SAMPLES_UE4 configured.")
        print("     Fire Z is estimated from its editor position, which may be wrong")
        print("     at different XY locations on the mountain.")
        print("     To fix: measure a few terrain Z values in UE4 editor and add them")
        print("     to TERRAIN_HEIGHT_SAMPLES_UE4 in fire_placer.py.")

    return {
        "actor": fire_name,
        "ue4_x": rand_x,
        "ue4_y": rand_y,
        "ue4_z": terrain_z_ue4,
        "airsim_x": ax,
        "airsim_y": ay,
        "airsim_z": az,
    }


def list_actors(client):
    """Just list fire actors and their current positions."""
    actors = find_fire_actors(client)
    print(f"\n  Fire actors in scene: {len(actors)}")
    for name in actors:
        try:
            pose = client.simGetObjectPose(name)
            ue4 = airsim_to_ue4(pose.position.x_val, pose.position.y_val, pose.position.z_val)
            print(f"    {name}")
            print(f"      AirSim : ({pose.position.x_val:.2f}, {pose.position.y_val:.2f}, {pose.position.z_val:.2f}) m")
            print(f"      UE4    : {ue4} cm")
        except Exception as e:
            print(f"    {name} → ❌ {e}")


def main():
    parser = argparse.ArgumentParser(description="Fire Placer — Random fire placement in AirSim")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for repeatable placement (omit for random)")
    parser.add_argument("--list", action="store_true",
                        help="List all fire actors and their positions, then exit")
    args = parser.parse_args()

    print("  🔌 Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("  ✅ Connected!")

    if args.list:
        list_actors(client)
        return

    result = place_fire(client, seed=args.seed)

    if result:
        print("\n" + "=" * 55)
        print("  📊 PLACEMENT SUMMARY")
        print("=" * 55)
        print(f"  Actor    : {result['actor']}")
        print(f"  UE4      : ({result['ue4_x']:.0f}, {result['ue4_y']:.0f}, {result['ue4_z']:.0f}) cm")
        print(f"  AirSim   : ({result['airsim_x']:.2f}, {result['airsim_y']:.2f}, {result['airsim_z']:.2f}) m")
        print("\n  ▶️  Next: python scripts/ai_tower_monitor.py")


if __name__ == "__main__":
    main()
