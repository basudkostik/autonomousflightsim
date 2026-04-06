"""
Map Generator Test & Verification
===================================
BlockingVolume, ağaç ve ateş Actor isimlerini doğrular.
Seed tekrarlanabilirliğini ve minimum mesafe kuralını test eder.

USAGE:
  python scripts/test_map_generator.py

PREREQUISITES:
  UE4 + AirSim running with Play pressed
"""

import airsim
import math
import time
from map_generator import MapGenerator, FIRE_NAME, VOLUME_NAME


def test_actor_names(client):
    """Sahnedeki Actor isimlerinin doğruluğunu kontrol et."""
    print("\n" + "=" * 50)
    print("  TEST 1: Actor Name Verification")
    print("=" * 50)

    objects = client.simListSceneObjects()
    passed = True

    # Ağaçlar
    tree_actors = [obj for obj in objects if obj.lower().startswith("tree_")]
    if not tree_actors:
        print(f"  ❌ FAIL — No tree actors found")
        passed = False
    else:
        print(f"  ✅ PASS — {len(tree_actors)} trees found")

    # Ateş
    if FIRE_NAME in objects:
        print(f"  ✅ PASS — Fire '{FIRE_NAME}' found")
    else:
        print(f"  ❌ FAIL — Fire '{FIRE_NAME}' not found")
        fire_candidates = [o for o in objects if "fire" in o.lower()]
        print(f"     Candidates: {fire_candidates}")
        passed = False

    # Volume
    if VOLUME_NAME in objects:
        print(f"  ✅ PASS — Landscape '{VOLUME_NAME}' found")
    else:
        print(f"  ❌ FAIL — Landscape '{VOLUME_NAME}' not found")
        vol_candidates = [o for o in objects if "landscape" in o.lower()]
        print(f"     Candidates: {vol_candidates}")
        passed = False

    return passed


def test_seed_reproducibility(client):
    """Aynı seed ile iki kez generate() → aynı pozisyonlar mı?"""
    print("\n" + "=" * 50)
    print("  TEST 2: Seed Reproducibility")
    print("=" * 50)

    seed = 12345

    # İlk generate
    gen1 = MapGenerator(client, seed=seed)
    gen1.generate()
    positions1 = gen1.get_tree_positions()
    fire1 = gen1.get_fire_position()

    time.sleep(1)

    # Aynı seed ile ikinci generate
    gen2 = MapGenerator(client, seed=seed)
    gen2.generate()
    positions2 = gen2.get_tree_positions()
    fire2 = gen2.get_fire_position()

    # Karşılaştır
    trees_match = True
    for i, (p1, p2) in enumerate(zip(positions1, positions2)):
        if abs(p1[0] - p2[0]) > 0.01 or abs(p1[1] - p2[1]) > 0.01:
            print(f"  ❌ Tree {i} mismatch: {p1} vs {p2}")
            trees_match = False

    fire_match = abs(fire1[0] - fire2[0]) < 0.01 and abs(fire1[1] - fire2[1]) < 0.01

    if trees_match and fire_match:
        print(f"  ✅ PASS — Seed {seed} produces identical positions")
    else:
        print(f"  ❌ FAIL — Positions differ for same seed!")

    return trees_match and fire_match


def test_different_seeds(client):
    """Farklı seed'ler → farklı pozisyonlar mı?"""
    print("\n" + "=" * 50)
    print("  TEST 3: Different Seeds → Different Maps")
    print("=" * 50)

    gen1 = MapGenerator(client, seed=111)
    gen1.generate()
    fire1 = gen1.get_fire_position()

    gen2 = MapGenerator(client, seed=222)
    gen2.generate()
    fire2 = gen2.get_fire_position()

    dist = math.sqrt((fire1[0] - fire2[0]) ** 2 + (fire1[1] - fire2[1]) ** 2)

    if dist > 1.0:
        print(f"  ✅ PASS — Fire moved {dist:.1f}m between seeds")
    else:
        print(f"  ⚠️  WARN — Fire only moved {dist:.1f}m (might be coincidence)")

    return dist > 1.0


def test_minimum_distance(client):
    """Tüm ağaç çiftleri arası mesafe ≥ MIN_TREE_DISTANCE mi?"""
    print("\n" + "=" * 50)
    print("  TEST 4: Minimum Tree Distance")
    print("=" * 50)

    gen = MapGenerator(client, seed=42)
    gen.generate()
    positions = gen.get_tree_positions()

    min_dist_found = float('inf')
    violations = 0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = math.sqrt(
                (positions[i][0] - positions[j][0]) ** 2 +
                (positions[i][1] - positions[j][1]) ** 2
            )
            min_dist_found = min(min_dist_found, d)
            if d < 7.9:  # Küçük tolerans
                violations += 1

    if violations == 0:
        print(f"  ✅ PASS — Min distance: {min_dist_found:.2f}m (no violations)")
    else:
        print(f"  ❌ FAIL — {violations} distance violations, min: {min_dist_found:.2f}m")

    return violations == 0


def test_bounds(client):
    """Tüm objeler BlockingVolume sınırları içinde mi?"""
    print("\n" + "=" * 50)
    print("  TEST 5: Bounds Check")
    print("=" * 50)

    gen = MapGenerator(client, seed=42)
    gen.generate()

    positions = gen.get_tree_positions()
    fire = gen.get_fire_position()
    bounds = gen.bounds

    out_of_bounds = 0
    for i, (x, y) in enumerate(positions):
        if x < bounds['x_min'] or x > bounds['x_max'] or y < bounds['y_min'] or y > bounds['y_max']:
            out_of_bounds += 1

    # Ateş de kontrol
    fx, fy = fire
    fire_ok = (bounds['x_min'] <= fx <= bounds['x_max'] and
               bounds['y_min'] <= fy <= bounds['y_max'])

    if out_of_bounds == 0 and fire_ok:
        print(f"  ✅ PASS — All objects within bounds")
    else:
        print(f"  ❌ FAIL — {out_of_bounds} trees + {'1 fire' if not fire_ok else '0 fire'} out of bounds")

    return out_of_bounds == 0 and fire_ok


def main():
    print("=" * 55)
    print("  🧪 MAP GENERATOR TEST SUITE")
    print("=" * 55)

    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!\n")

    results = {}
    results["Actor Names"] = test_actor_names(client)
    results["Seed Reproducibility"] = test_seed_reproducibility(client)
    results["Different Seeds"] = test_different_seeds(client)
    results["Minimum Distance"] = test_minimum_distance(client)
    results["Bounds Check"] = test_bounds(client)

    # Sonuç özeti
    print("\n" + "=" * 55)
    print("  📊 TEST RESULTS")
    print("=" * 55)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")
        if not passed:
            all_passed = False

    print("=" * 55)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print("  ⚠️  Some tests failed. Check output above.")
    print("=" * 55)


if __name__ == "__main__":
    main()
