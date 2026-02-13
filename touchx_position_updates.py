import time
from dataclasses import dataclass

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice


# Spatial boundaries (in mm)
BOUNDARY_MIN_X = -100.0
BOUNDARY_MAX_X = 100.0
BOUNDARY_MIN_Y = -100.0
BOUNDARY_MAX_Y = 100.0
BOUNDARY_MIN_Z = -100.0
BOUNDARY_MAX_Z = 100.0

# Force feedback parameters
BOUNDARY_FORCE_GAIN = 0.3  # Force gain (N/mm) - how strong the force is per mm exceeded
MAX_BOUNDARY_FORCE = 2.0   # Maximum force (N)


# Shared state object
@dataclass
class DeviceState:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    out_of_bounds: bool = False  # Track if currently out of bounds


device_state = DeviceState()

# Callback function; runs in servo thread 
@hd_callback
def device_callback():
    global device_state

    # Get current position from transform
    transform = hd.get_transform()
    current_position = [transform[3][0], -transform[3][1], transform[3][2]]
    
    # Update state
    device_state.last_position = device_state.position
    device_state.position = current_position
    
    x, y, z = current_position
    
    '''
    # Calculate force based on boundary violations
    force_x, force_y, force_z = 0.0, 0.0, 0.0
    
    # Check X boundary
    if x < BOUNDARY_MIN_X:
        penetration = BOUNDARY_MIN_X - x
        force_x = min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    elif x > BOUNDARY_MAX_X:
        penetration = x - BOUNDARY_MAX_X
        force_x = -min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    
    # Check Y boundary
    if y < BOUNDARY_MIN_Y:
        penetration = BOUNDARY_MIN_Y - y
        force_y = min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    elif y > BOUNDARY_MAX_Y:
        penetration = y - BOUNDARY_MAX_Y
        force_y = -min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    
    # Check Z boundary
    if z < BOUNDARY_MIN_Z:
        penetration = BOUNDARY_MIN_Z - z
        force_z = min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    elif z > BOUNDARY_MAX_Z:
        penetration = z - BOUNDARY_MAX_Z
        force_z = -min(penetration * BOUNDARY_FORCE_GAIN, MAX_BOUNDARY_FORCE)
    
    # Apply force feedback if any boundary is exceeded
    if force_x != 0.0 or force_y != 0.0 or force_z != 0.0:
        hd.set_force([force_x, force_y, force_z])
        # Log when transitioning from in-bounds to out-of-bounds
        if not device_state.out_of_bounds:
            print(f"\nOut of bounds detected at ({x:.2f}, {y:.2f}, {z:.2f}) mm - Applying force ({force_x:.2f}, {force_y:.2f}, {force_z:.2f}) N")
            device_state.out_of_bounds = True
    else:
        hd.set_force([0.0, 0.0, 0.0])
        # Reset flag when back in bounds
        if device_state.out_of_bounds:
            print(f"\nBack in bounds at ({x:.2f}, {y:.2f}, {z:.2f}) mm")
            device_state.out_of_bounds = False

    # return hd.HD_CALLBACK_CONTINUE # returns continue flag so callback loop can continue
    '''


def main():
    device = HapticDevice(callback=device_callback, scheduler_type="async") # Initialize device and set callback

    try:
        print("Reading positions from Touch X")
        #print(f"Boundaries: X[{BOUNDARY_MIN_X}, {BOUNDARY_MAX_X}], "
       #       f"Y[{BOUNDARY_MIN_Y}, {BOUNDARY_MAX_Y}], "
       #       f"Z[{BOUNDARY_MIN_Z}, {BOUNDARY_MAX_Z}] mm\n")
        while True:
            # Read latest updated position from state
            x, y, z = device_state.position
            
            '''
            # Check if any boundary is exceeded
            boundary_exceeded = (x < BOUNDARY_MIN_X or x > BOUNDARY_MAX_X or
                               y < BOUNDARY_MIN_Y or y > BOUNDARY_MAX_Y or
                               z < BOUNDARY_MIN_Z or z > BOUNDARY_MAX_Z)
            
            status = " [BOUNDARY!]" if boundary_exceeded else ""
            '''
            #print(f"Position (mm): x={x:7.2f}, y={y:7.2f}, z={z:7.2f}{status}    ", end="\r")
            
            print(f"Position (mm): x={x:7.2f}, y={y:7.2f}, z={z:7.2f}    ", end="\r")
            time.sleep(0.02)  # Thread runs at 1kHz but including delay
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Close device
        device.close()


if __name__ == "__main__":
    main()