"""
Seed-Based Procedural Map Generator
====================================
Her episode başında ağaçları ve ateşi rastgele konumlara yerleştirir.
BlockingVolume sınırlarını AirSim API'den okuyarak harita sınırlarını belirler.

Amaç: RL ajanının haritayı ezberlemesini önlemek —
      aynı seed = aynı harita, farklı seed = farklı harita.

USAGE:
  # Modül olarak kullanım (ppo_environment.py reset() içinde):
  from map_generator import MapGenerator
  map_gen = MapGenerator(client, seed=42)
  map_gen.generate()

  # Bağımsız test:
  python scripts/map_generator.py --seed 42

PREREQUISITES:
  1. UE4 sahnesinde 106 ağaç Actor'ü: tree_0 .. tree_105
  2. Ateş Actor'ü: Fire_15_Blueprint
  3. BlockingVolume Actor'ü harita sınırlarını tanımlar
  4. UE4 running with Play pressed
"""

import airsim
import random
import math
import argparse
import time


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
FIRE_NAME = "Fire_15_Blueprint_2"         # UE4'teki ateş Actor ismi
VOLUME_NAME = "Landscape_0"               # Harita sınırlarını belirleyen volume

# Landscape Scale değeri genelde (100,100,100) gelir, quad/component sayısını içermez.
# Bu yüzden harita boyutu on binlerce metre hesaplanıp ağaçlar harita dışına uçar (Kill Z sebebiyle silinir).
# Kullanıcının okuduğu UE4 köşe koordinatlarını (cm) AirSim metre cinsine dönüştüreceğiz.
USE_MANUAL_BOUNDS = True
MANUAL_BOUNDS_UE4_CM = {
    'x_min': -17210.0,
    'x_max': 11920.0,
    'y_min': -7060.0,
    'y_max': 22140.0
}

# AirSim koordinat sistemi (0,0) noktası Drone'un başlangıç noktasıdır (PlayerStart).
# Eğer Drone'un haritadaki başlangıç pozisyonu değiştirilirse, AirSim'in kaymasını 
# engellemek için Drone'un Unreal Editor'deki mutlak (World) konumunu (cm) buraya girin.
DRONE_START_UE4_CM = {
    'x': -2500.0,
    'y': 10000.0
}

MIN_TREE_DISTANCE = 8.0                # Ağaçlar arası minimum mesafe (metre)
FIRE_MARGIN = 20.0                     # Ateş kenarlardan bu kadar içeride olacak (metre)
TREE_Z = 0.0                           # Ağaçların Z pozisyonu (zemin)
FIRE_Z = 0.0                           # Ateş Z pozisyonu


class MapGenerator:
    """
    Seed tabanlı harita oluşturucu.

    BlockingVolume'dan sınırları okur, ardından tüm ağaç ve ateş
    Actor'lerini seed kontrollü rastgele pozisyonlara yerleştirir.
    """

    def __init__(self, client: airsim.MultirotorClient, seed: int = None):
        """
        Args:
            client: AirSim bağlantısı
            seed:   Rastgelelik tohumu. None = her seferinde farklı.
        """
        self.client = client
        self.seed = seed if seed is not None else random.randint(0, 999_999)
        self.rng = random.Random(self.seed)

        # Sahnedeki tüm tree Actor'lerini otomatik bul (tree_ veya Tree_)
        all_objects = client.simListSceneObjects()
        self.tree_names = sorted([
            obj for obj in all_objects
            if obj.lower().startswith("tree_")
        ])
        self.fire_name = FIRE_NAME

        print(f"  🌲 {len(self.tree_names)} tree actors found in scene")

        # Sınırlar — generate() çağrıldığında BlockingVolume'dan okunacak
        self.bounds = None

        # Yerleştirilen pozisyonlar (doğrulama için)
        self.tree_positions = []
        self.fire_position = None

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def generate(self):
        """
        Seed'e göre tüm objeleri yerleştir.
        Returns: kullanılan seed değeri
        """
        print(f"\n{'═' * 50}")
        print(f"  🗺️  MAP GENERATOR — Seed: {self.seed}")
        print(f"{'═' * 50}")

        # RNG'yi sıfırla (aynı seed = aynı sonuç garantisi)
        self.rng = random.Random(self.seed)

        # 1. BlockingVolume'dan sınırları oku
        self._read_bounds()

        # 2. Ağaçları yerleştir
        self._place_trees()

        # 3. Ateşi yerleştir
        self._place_fire()

        print(f"\n  ✅ Map generated successfully!")
        print(f"  📊 {len(self.tree_positions)} trees + 1 fire placed")
        print(f"  🔑 Seed: {self.seed} (bu seed'i kullanarak aynı haritayı tekrar oluşturabilirsiniz)")
        print(f"{'═' * 50}\n")

        return self.seed

    def get_fire_position(self):
        """Ateş pozisyonunu döndür (x, y). Reward hesabı için kullanılır."""
        if self.fire_position:
            return self.fire_position
        pose = self.client.simGetObjectPose(self.fire_name)
        return (pose.position.x_val, pose.position.y_val)

    def get_tree_positions(self):
        """Tüm ağaç pozisyonlarını döndür [(x, y), ...]."""
        return list(self.tree_positions)

    # ─────────────────────────────────────────────
    # INTERNAL METHODS
    # ─────────────────────────────────────────────

    def _read_bounds(self):
        """
        Landscape Actor'ünün pozisyon ve scale'inden harita sınırlarını hesapla.

        UE4'te objenin merkez pozisyonu ve scale'i haritanın sınırlarını belirler.
        AirSim bu bilgiyi simGetObjectPose() ve simGetObjectScale() ile verir.
        Ancak Landscape objelerinde boyut Scale ile doğrudan orantılı değildir.
        """
        if USE_MANUAL_BOUNDS:
            # UE4 cm (Dünya Koordinatları) -> AirSim metre (NED)
            # 100 birim = 1 metre
            # Drone'un başlangıç pozisyonundan çıkartarak (offset) AirSim sistemine uyarlıyoruz
            self.bounds = {
                'x_min': (MANUAL_BOUNDS_UE4_CM['x_min'] - DRONE_START_UE4_CM['x']) / 100.0,
                'x_max': (MANUAL_BOUNDS_UE4_CM['x_max'] - DRONE_START_UE4_CM['x']) / 100.0,
                'y_min': (MANUAL_BOUNDS_UE4_CM['y_min'] - DRONE_START_UE4_CM['y']) / 100.0,
                'y_max': (MANUAL_BOUNDS_UE4_CM['y_max'] - DRONE_START_UE4_CM['y']) / 100.0,
            }
            print(f"  📦 Using Manual Landscape bounds (meters) [Offset applied]:")
            print(f"     X: [{self.bounds['x_min']:.1f}, {self.bounds['x_max']:.1f}]")
            print(f"     Y: [{self.bounds['y_min']:.1f}, {self.bounds['y_max']:.1f}]")
            return

        try:
            pose = self.client.simGetObjectPose(VOLUME_NAME)
            scale = self.client.simGetObjectScale(VOLUME_NAME)

            # Volume merkezi
            cx = pose.position.x_val
            cy = pose.position.y_val

            # Scale → boyut (UE4'te default box 200x200x200 cm = 2x2x2 m)
            # Scale * 100 = yarım boyut (cm cinsinden), / 100 = metre
            # AirSim ile UE4 arasındaki dönüşüm: 1 AirSim birimi = 1 metre = 100 UE4 birimi
            half_x = scale.x_val * 100  # Scale genelde 1 birim = 100cm
            half_y = scale.y_val * 100

            self.bounds = {
                'x_min': cx - half_x,
                'x_max': cx + half_x,
                'y_min': cy - half_y,
                'y_max': cy + half_y,
            }

            print(f"  📦 Landscape bounds:")
            print(f"     X: [{self.bounds['x_min']:.1f}, {self.bounds['x_max']:.1f}]")
            print(f"     Y: [{self.bounds['y_min']:.1f}, {self.bounds['y_max']:.1f}]")

        except Exception as e:
            print(f"  ⚠️  Landscape okunamadı: {e}")
            print(f"  ⚠️  Varsayılan sınırlar kullanılıyor: ±100m")
            self.bounds = {
                'x_min': -100, 'x_max': 100,
                'y_min': -100, 'y_max': 100,
            }

    def _place_trees(self):
        """
        Ağaçları minimum mesafe kuralıyla rastgele yerleştir.

        Algoritma:
        1. Her ağaç için bounds içinde rastgele bir (x, y) seç
        2. Daha önce yerleştirilen ağaçlarla minimum mesafe kontrolü yap
        3. Çakışma varsa yeni pozisyon dene (max 100 deneme)
        4. simSetObjectPose ile AirSim'de objeyi taşı
        """
        self.tree_positions = []
        placed = 0
        failed = 0

        print(f"\n  🌲 Placing {len(self.tree_names)} trees (min distance: {MIN_TREE_DISTANCE}m)...")

        for name in self.tree_names:
            pos = self._find_valid_position(min_dist=MIN_TREE_DISTANCE)

            if pos:
                x, y = pos
                self.tree_positions.append((x, y))

                # Rastgele rotasyon (yaw only — doğal görünsün)
                yaw_deg = self.rng.uniform(0, 360)
                yaw_rad = math.radians(yaw_deg)
                quat = self._yaw_to_quaternion(yaw_rad)

                # Ağacın kendi Z koordinatını koru (drone'un yüksekliğinde havada değil zeminde kalması için)
                current_pose = self.client.simGetObjectPose(name)
                current_z = current_pose.position.z_val if not math.isnan(current_pose.position.z_val) else TREE_Z

                pose = airsim.Pose(
                    airsim.Vector3r(x, y, current_z),
                    quat
                )
                success = self.client.simSetObjectPose(name, pose)

                if success:
                    placed += 1
                    if placed <= 10 or placed == len(self.tree_names): # İlk 10'unu ve sonuncusunu yazdır kalabalık olmasın diye
                        print(f"     [Tree {placed:03d}] {name} -> X: {x:.1f}, Y: {y:.1f}, Z: {current_z:.1f}")
                else:
                    failed += 1
            else:
                failed += 1

        print(f"     ✅ Placed: {placed} | ⚠️ Failed: {failed}")

    def _place_fire(self):
        """
        Ateşi sınırların iç kısmına rastgele yerleştir.

        Kenarlardan FIRE_MARGIN kadar içeride kalır,
        böylece drone her zaman ateşe ulaşabilir.
        """
        print(f"\n  🔥 Placing fire...")

        x = self.rng.uniform(
            self.bounds['x_min'] + FIRE_MARGIN,
            self.bounds['x_max'] - FIRE_MARGIN
        )
        y = self.rng.uniform(
            self.bounds['y_min'] + FIRE_MARGIN,
            self.bounds['y_max'] - FIRE_MARGIN
        )

        # Ateşin mevcut Z koordinatını koruyalım
        current_pose = self.client.simGetObjectPose(self.fire_name)
        current_z = current_pose.position.z_val if not math.isnan(current_pose.position.z_val) else FIRE_Z

        pose = airsim.Pose(
            airsim.Vector3r(x, y, current_z),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        success = self.client.simSetObjectPose(self.fire_name, pose)
        self.fire_position = (x, y)

        if success:
            print(f"     ✅ Fire '{self.fire_name}' placed at -> X: {x:.1f}, Y: {y:.1f}, Z: {current_z:.1f}")
        else:
            print(f"     ❌ Failed to place fire! Check actor name: '{self.fire_name}'")

    def _find_valid_position(self, min_dist: float, max_attempts: int = 100):
        """
        Mevcut ağaçlarla çakışmayan rastgele bir pozisyon bul.

        Returns: (x, y) tuple veya None (başarısız)
        """
        for _ in range(max_attempts):
            x = self.rng.uniform(self.bounds['x_min'], self.bounds['x_max'])
            y = self.rng.uniform(self.bounds['y_min'], self.bounds['y_max'])

            # Daha önce yerleştirilen tüm ağaçlarla mesafe kontrolü
            too_close = False
            for (px, py) in self.tree_positions:
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < min_dist:
                    too_close = True
                    break

            # Drone Keep-Out Zone: ensure trees don't spawn on top of drone spawn (0,0)
            if math.sqrt(x**2 + y**2) < 15.0:
                too_close = True

            if not too_close:
                return (x, y)

        return None

    @staticmethod
    def _yaw_to_quaternion(yaw_rad: float) -> airsim.Quaternionr:
        """Yaw açısını (radyan) quaternion'a çevir."""
        return airsim.Quaternionr(
            0,
            0,
            math.sin(yaw_rad / 2),
            math.cos(yaw_rad / 2)
        )


# ═══════════════════════════════════════════════════════
# STANDALONE EXECUTION — Test & Debug
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Seed-based map generator for RL training")
    parser.add_argument("--seed", type=int, default=None,
                        help="Map seed (default: random)")
    parser.add_argument("--list-objects", action="store_true",
                        help="List all scene objects and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Verify tree and fire actor names exist in scene")
    args = parser.parse_args()

    # AirSim bağlantısı
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!\n")

    # ── Obje listesi modu ──
    if args.list_objects:
        print("📋 Scene objects:")
        objects = client.simListSceneObjects()
        for obj in sorted(objects):
            print(f"   {obj}")
        print(f"\nTotal: {len(objects)} objects")
        return

    # ── Doğrulama modu ──
    if args.verify:
        print("🔍 Verifying actor names...\n")
        objects = client.simListSceneObjects()
        objects_lower = {o.lower(): o for o in objects}

        # Ağaçları kontrol et
        missing_trees = []
        # Sahnedeki ağaç isimlerini list_objects veya __init__'ten dinamik çekmek daha iyi ancak main() içinde args.verify kısmındayız.
        # Tree_ ve tree_ prefixli tüm objeleri bulalım:
        tree_actors = [obj for obj in objects if obj.lower().startswith("tree_")]
        
        if not tree_actors:
            print(f"  ❌ No tree actors found starting with 'tree_' or 'Tree_'!")
        else:
            print(f"  ✅ Found {len(tree_actors)} tree actors in scene.")

        # Ateşi kontrol et
        if FIRE_NAME in objects or FIRE_NAME.lower() in objects_lower:
            print(f"  ✅ Fire actor '{FIRE_NAME}' found!")
        else:
            print(f"  ❌ Fire actor '{FIRE_NAME}' NOT found!")
            # Benzer isim öner
            suggestions = [o for o in objects if "fire" in o.lower()]
            if suggestions:
                print(f"     Possible matches: {suggestions}")

        # Volume kontrol et
        if VOLUME_NAME in objects or VOLUME_NAME.lower() in objects_lower:
            print(f"  ✅ Volume '{VOLUME_NAME}' found!")
            pose = client.simGetObjectPose(VOLUME_NAME)
            print(f"     Position: ({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f})")
        else:
            print(f"  ❌ Volume '{VOLUME_NAME}' NOT found!")
            suggestions = [o for o in objects if "volume" in o.lower() or "blocking" in o.lower()]
            if suggestions:
                print(f"     Possible matches: {suggestions}")

        return

    # ── Harita oluşturma modu ──
    map_gen = MapGenerator(client, seed=args.seed)
    used_seed = map_gen.generate()

    # Sonuç özeti
    fire_pos = map_gen.get_fire_position()
    print(f"\n📊 Summary:")
    print(f"   Seed:          {used_seed}")
    print(f"   Trees placed:  {len(map_gen.tree_positions)}")
    print(f"   Fire position: ({fire_pos[0]:.1f}, {fire_pos[1]:.1f})")


if __name__ == "__main__":
    main()
