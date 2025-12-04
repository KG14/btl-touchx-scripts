import time
from dataclasses import dataclass

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice


# Shared state object
@dataclass
class DeviceState:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


device_state = DeviceState()

# Callback function; runs in servo thread 
@hd_callback
def device_callback():
    global device_state

    # device = hd.get_current_device()
    transform = hd.get_transform()
    device_state.position = [transform[3][0], -transform[3][1], transform[3][2]]

    # return hd.HD_CALLBACK_CONTINUE # returns continue flag so callback loop can continue


def main():
    device = HapticDevice(callback=device_callback, scheduler_type="async") # Initialize device and set callback

    try:
        print("Reading positions from Touch X\n")
        while True:
            # Read latest updated position from state
            x, y, z = device_state.position
            print(f"Position (mm): x={x:7.2f}, y={y:7.2f}, z={z:7.2f}", end="\r")
            time.sleep(0.02)  # Thread runs at 1kHz but including delay
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Close device
        device.close()


if __name__ == "__main__":
    main()