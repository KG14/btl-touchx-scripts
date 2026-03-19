import time
from dataclasses import dataclass, field

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice

# Force feedback parameters
X_THRESHOLD = 80.0   # mm — force activates beyond this x value
STIFFNESS = 0.3      # N/mm — spring gain
DAMPING = 0.002      # N·s/mm — velocity damping to prevent whipping
MAX_FORCE = 2.0      # N — safety clamp
SERVO_DT = 0.001     # s — servo loop runs at ~1kHz


@dataclass
class DeviceState:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    prev_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force_active: bool = False
    applied_force: float = 0.0


device_state = DeviceState()


@hd_callback
def device_callback():
    global device_state

    transform = hd.get_transform()
    pos = [transform[3][0], transform[3][1], transform[3][2]]

    vel_x = (pos[0] - device_state.prev_position[0]) / SERVO_DT

    device_state.prev_position = device_state.position
    device_state.position = pos

    x = pos[0]

    if x > X_THRESHOLD:
        penetration = x - X_THRESHOLD
        force_x = -STIFFNESS * penetration - DAMPING * vel_x
        force_x = max(-MAX_FORCE, min(MAX_FORCE, force_x))
        hd.set_force([force_x, 0.0, 0.0])
        device_state.force_active = True
        device_state.applied_force = force_x
    else:
        if device_state.force_active:
            hd.set_force([0.0, 0.0, 0.0])
            device_state.force_active = False
            device_state.applied_force = 0.0


def main():
    device = HapticDevice(callback=device_callback, scheduler_type="async")

    try:
        print("Tracking Touch X position — force boundary at x > "
              f"{X_THRESHOLD} mm")
        print(f"Stiffness={STIFFNESS} N/mm, Damping={DAMPING} N·s/mm, "
              f"Max={MAX_FORCE} N\n")

        while True:
            x, y, z = device_state.position
            force_status = (f" [FORCE: {device_state.applied_force:+.3f} N]"
                           if device_state.force_active else "")
            print(f"Position (mm): x={x:7.2f}, y={y:7.2f}, z={z:7.2f}"
                  f"{force_status:30s}", end="\r")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        device.close()


if __name__ == "__main__":
    main()
