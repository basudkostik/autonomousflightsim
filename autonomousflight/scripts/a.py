import airsim
import time

def print_state(client, label="STATE"):
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity

    print(f"\n🔍 {label}")
    print(f"  Position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
    print(f"  Velocity: VX={vel.x_val:.2f}, VY={vel.y_val:.2f}, VZ={vel.z_val:.2f}")
    print(f"  Landed State: {state.landed_state}")
    print(f"  Armed: {client.isApiControlEnabled()}")


def debug_drone():
    print("🔌 Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    print("✅ Connected!")

    # RESET
    print("\n🔄 Resetting drone...")
    client.reset()
    time.sleep(1)

    # ENABLE API
    print("🎮 Enabling API Control...")
    client.enableApiControl(True)

    # ARM
    print("🔐 Arming drone...")
    client.armDisarm(True)

    print_state(client, "INITIAL STATE")

    # COLLISION CHECK
    collision = client.simGetCollisionInfo()
    print(f"\n💥 Collision: {collision.has_collided}")

    # TAKEOFF TEST
    print("\n🛫 TAKEOFF TEST...")
    client.takeoffAsync().join()
    time.sleep(2)

    print_state(client, "AFTER TAKEOFF")

    # ALTITUDE TEST
    print("\n📡 ALTITUDE TEST (moveToZAsync -10)...")
    client.moveToZAsync(-10, 3).join()
    time.sleep(2)

    print_state(client, "AFTER moveToZAsync")

    # VELOCITY TEST
    print("\n➡️ VELOCITY TEST (forward)...")
    client.moveByVelocityAsync(3, 0, 0, duration=3).join()

    print_state(client, "AFTER VELOCITY")

    # MANUAL Z TEST (CRITICAL)
    print("\n⬆️ MANUAL Z TEST (force upward)...")
    for i in range(5):
        client.moveByVelocityAsync(0, 0, -2, duration=1)
        time.sleep(1)
        print_state(client, f"Z TEST {i+1}")

    print("\n🛑 Landing...")
    client.landAsync().join()

    client.armDisarm(False)
    client.enableApiControl(False)

    print("\n✅ DEBUG COMPLETE")


if __name__ == "__main__":
    debug_drone()