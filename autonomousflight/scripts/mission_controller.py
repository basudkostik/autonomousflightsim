"""
Mission Controller — Full 3-Phase Integration
================================================
Orchestrates the complete fire detection and verification mission:

  Phase A → Bearing Navigation (High Altitude)
  Phase B → Plume Tracking (Mid Altitude, Descent)
  Phase C → PPO Navigation (Low Altitude, Under Canopy)

USAGE:
  python scripts/mission_controller.py --bearing 127.0
  python scripts/mission_controller.py --bearing 45.0 --model models/ppo_forest_final.zip

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Fire/smoke actor in the scene
  3. Trained PPO model (for Phase C)
"""

import airsim
import os
import sys
import time
import json
import argparse

# Fix Windows console encoding for emoji output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from bearing_navigator import BearingNavigator
from plume_tracker import PlumeTracker
from ppo_navigator import PPONavigator


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "ppo_forest_final.zip"
)


class MissionController:
    """
    Orchestrates the full 3-phase fire approach mission.

    States:
        IDLE → TAKEOFF → BEARING_NAV → PLUME_TRACK → PPO_NAV → REPORT → RETURN
    """

    def __init__(self, client, model_path=DEFAULT_MODEL_PATH):
        """
        Args:
            client: airsim.MultirotorClient (already connected)
            model_path: Path to trained PPO model for Phase C
        """
        self.client = client
        self.model_path = model_path
        self.state = "IDLE"
        self.mission_log = {
            "start_time": None,
            "phases": {},
            "result": None
        }

        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "mission")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_position(self):
        """Get current drone position."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def takeoff(self):
        """Prepare drone for mission."""
        self.state = "TAKEOFF"
        print("\n" + "🟢" * 30)
        print("  🚁 MISSION START — AUTONOMOUS FIRE DETECTION")
        print("🟢" * 30)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print("  🛫 Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)

        # Initial climb
        self.client.moveToZAsync(-20, 5).join()
        time.sleep(1)

        print("  ✅ Airborne and ready!")

    def run_phase_a(self, bearing):
        """
        Execute Phase A: Bearing Navigation.

        Args:
            bearing: Target bearing angle from tower

        Returns:
            dict: Phase A result
        """
        self.state = "BEARING_NAV"
        print(f"\n{'─' * 55}")
        print(f"  ENTERING PHASE A — Bearing: {bearing:.1f}°")
        print(f"{'─' * 55}")

        nav = BearingNavigator(self.client, bearing)
        result = nav.fly_toward_bearing()

        self.mission_log["phases"]["A"] = {
            "result": result,
            "flight_log_entries": len(nav.flight_log)
        }

        return result

    def run_phase_b(self):
        """
        Execute Phase B: Plume Tracking.

        Returns:
            dict: Phase B result with estimated fire position
        """
        self.state = "PLUME_TRACK"
        print(f"\n{'─' * 55}")
        print(f"  ENTERING PHASE B — Plume Tracking")
        print(f"{'─' * 55}")

        tracker = PlumeTracker(self.client)
        result = tracker.track_plume()

        self.mission_log["phases"]["B"] = {
            "result": result,
            "flight_log_entries": len(tracker.flight_log)
        }

        return result

    def run_phase_c(self, fire_estimate):
        """
        Execute Phase C: PPO Navigation.

        Args:
            fire_estimate: (x, y) estimated fire position from Phase B

        Returns:
            dict: Phase C result with fire confirmation
        """
        self.state = "PPO_NAV"
        print(f"\n{'─' * 55}")
        print(f"  ENTERING PHASE C — PPO Under-Canopy Navigation")
        print(f"{'─' * 55}")

        if not os.path.exists(self.model_path):
            print(f"  ⚠️ PPO model not found at: {self.model_path}")
            print(f"  ⚠️ Skipping Phase C — run train_ppo.py first!")
            return {
                "fire_confirmed": False,
                "reason": "model_not_found",
                "position": self.get_position()
            }

        nav = PPONavigator(self.client, self.model_path, fire_estimate)
        result = nav.navigate_and_confirm()

        self.mission_log["phases"]["C"] = {
            "result": result,
            "flight_log_entries": len(nav.flight_log)
        }

        return result

    def return_to_base(self):
        """Return drone to starting position and land."""
        self.state = "RETURN"
        print(f"\n{'─' * 55}")
        print(f"  🔙 RETURNING TO BASE")
        print(f"{'─' * 55}")

        # Climb to safe altitude first
        print("  📡 Climbing to safe altitude...")
        self.client.moveToZAsync(-10, 5).join()
        time.sleep(1)

        # Return to origin
        print("  🏠 Flying back to origin...")
        self.client.moveToPositionAsync(0, 0, -10, 8).join()
        time.sleep(1)

        # Land
        print("  🔽 Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("  ✅ Landed safely!")

    def report_mission(self, fire_confirmed, final_position):
        """Generate mission report."""
        self.state = "REPORT"
        elapsed = time.time() - self.mission_log["start_time"]

        report = {
            "mission_time": elapsed,
            "fire_confirmed": fire_confirmed,
            "final_position": final_position,
            "phases": self.mission_log["phases"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print("\n" + "=" * 55)
        print("  📋 MISSION REPORT")
        print("=" * 55)
        print(f"  🕐 Total time: {elapsed:.1f}s")
        print(f"  🔥 Fire confirmed: {'✅ YES' if fire_confirmed else '❌ NO'}")
        if final_position:
            x, y, z = final_position
            print(f"  📍 Fire location: ({x:.1f}, {y:.1f}, {z:.1f})")
        print("=" * 55)

        # Save report
        report_path = os.path.join(
            self.output_dir,
            f"mission_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  📝 Report saved: {report_path}")

        return report

    def run_mission(self, bearing):
        """
        Execute the complete 3-phase mission.

        Args:
            bearing: Target bearing angle from the tower

        Returns:
            dict: Mission report
        """
        self.mission_log["start_time"] = time.time()

        try:
            # ──────────────────────────────────────
            # TAKEOFF
            # ──────────────────────────────────────
            self.takeoff()

            # ──────────────────────────────────────
            # PHASE A: Bearing Navigation
            # ──────────────────────────────────────
            phase_a_result = self.run_phase_a(bearing)

            if phase_a_result["reason"] == "timeout":
                print("  ⚠️ Phase A timed out — attempting Phase B anyway")

            # ──────────────────────────────────────
            # PHASE B: Plume Tracking
            # ──────────────────────────────────────
            phase_b_result = self.run_phase_b()

            # Estimate fire position from Phase B result
            if "position" in phase_b_result:
                fire_estimate = (
                    phase_b_result["position"][0],
                    phase_b_result["position"][1]
                )
            else:
                # Use current position as fallback
                x, y, z = self.get_position()
                fire_estimate = (x, y)

            # ──────────────────────────────────────
            # PHASE C: PPO Navigation
            # ──────────────────────────────────────
            phase_c_result = self.run_phase_c(fire_estimate)

            fire_confirmed = phase_c_result.get("fire_confirmed", False)
            final_position = phase_c_result.get("position", self.get_position())

            # ──────────────────────────────────────
            # REPORT & RETURN
            # ──────────────────────────────────────
            report = self.report_mission(fire_confirmed, final_position)
            self.return_to_base()

            return report

        except Exception as e:
            print(f"\n❌ MISSION ABORT: {e}")
            import traceback
            traceback.print_exc()

            # Emergency: climb and land
            try:
                print("  🚨 EMERGENCY — Climbing to safe altitude...")
                self.client.moveToZAsync(-50, 10).join()
                time.sleep(2)
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except Exception:
                pass

            return {"error": str(e), "fire_confirmed": False}


def main():
    """Standalone mission execution."""
    parser = argparse.ArgumentParser(description="Mission Controller — Full 3-Phase Mission")
    parser.add_argument("--bearing", type=float, required=True,
                        help="Target bearing from the AI tower (degrees)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained PPO model")
    args = parser.parse_args()

    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!")

    controller = MissionController(client, args.model)
    report = controller.run_mission(args.bearing)

    print(f"\n🏁 Mission complete!")
    print(f"   Fire confirmed: {report.get('fire_confirmed', False)}")


if __name__ == "__main__":
    main()
