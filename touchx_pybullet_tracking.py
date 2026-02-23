import pybullet as p
import numpy as np
import pybullet_data
import pybullet_utils_cust as pu
import os
import time
from UR5Controller import UR5Controller

from dataclasses import dataclass
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice

'''
========= TouchX interface config ========
'''
# Shared device state object
@dataclass
class DeviceState:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
device_state = DeviceState() # Instantiate

# Callback function; runs in servo thread 
@hd_callback
def device_callback():
    global device_state

    # device = hd.get_current_device()
    transform = hd.get_transform()
    device_state.position = [transform[3][0], -transform[3][1], transform[3][2]]

'''
========= PyBullet Sim functions ==========
'''    
def load_ur_robot(robot_initial_pose=[[0., 0., 0.], [0., 0., 0., 1.]], 
                          client_id=0,urdf_dir="./robots/urdf/", urdf_file = "ur5e_fixed.urdf"):
    urdf_filepath = os.path.join(urdf_dir, urdf_file)  # Path to the URDF file, a file that contains information about the robot geometry and connections

    if not os.path.exists(urdf_filepath): # Check to make sure the path exists, and raise an error if it doesn't
        raise ValueError(f"URDF file {urdf_filepath} does not exist!")

    # Load the robot (by using the URDF file), and also assign it an initial position and angle for the base (basePosition and baseOrientation)
    robot_id = p.loadURDF(urdf_filepath, 
                          basePosition=robot_initial_pose[0], 
                          baseOrientation=robot_initial_pose[1],
                          useFixedBase=True,
                          physicsClientId=client_id,
                          #flags=p.URDF_USE_SELF_COLLISION  # Enable self-collision checking
                        )
    return robot_id, urdf_filepath

# Constants 
A = np.array([ # Axes transform matrix
    [1,  0,  0],  # pb_x = + touch_x
    [0,  0,  1],  # pb_y = + touch_z
    [0, -1,  0],  # pb_z = - touch_y
], dtype=float)
s = 0.001 # scale (TouchX mm -> PB meters)
p_pb_home = np.array([0, 0, 0]) # PB home position
target_orientation = np.array([0, 0, 0, 1]) # Constant target PB orientation for now

# Global variables
p_touchx_home = np.zeros(3) # TouchX starting position 
current_p_touchx = np.zeros(3) # Current TouchX position; continuously updated
current_p_sim = np.zeros(3) # Current PyBullet sim position; continuously updated
client_id, robot, robot_id = None, None, None
startup = True

# Function to convert TouchX position to PyBullet axes
def touchx_to_pb_pos(p):
    global p_touchx_home
    return s * (A @ (p - p_touchx_home))

# Function to find change in position & convert to PyBullet sim axes
def findPositionDelta(current, new):
    return s * (A @ (new - current))

# Function to move arm to a given position (x,y,z) and orientation (quaternion)
def move_to_position(target_position):
    global client_id, robot, robot_id

    ikJoints = p.calculateInverseKinematics(robot.id,
                                            8, # end-effector index
                                            targetPosition=target_position,
                                            maxNumIterations=2000,
                                            residualThreshold=1e-5,
                                            physicsClientId=client_id)
    
    # ************************ You don't need to understand this part below too much right now, though you should be able to get most of it just by reading the function and variable names.
    # ************************ Basically it just checks your joint solution to make sure it isn't outside robot limits, and it actually gets you close to the target position and orientation you wanted.
    within_limits = True
    arm_lowerLimits, arm_upperLimits = pu.get_joints_limits(robot_id, joints=robot.GROUP_INDEX['arm'], client_id=client_id)
    for idx, joint_value in enumerate(ikJoints):
        lower_limit = arm_lowerLimits[idx]
        upper_limit = arm_upperLimits[idx]
        if not (lower_limit <= joint_value <= upper_limit):
            print(f"Joint {idx} value {joint_value} is out of limits [{lower_limit}, {upper_limit}]")
            within_limits = False
            break
    if not within_limits:
        print("IK solution is outside joint limits.")
        #continue
    position_error, orientation_error = robot.check_position_feasibility(ikJoints, target_position, target_orientation, 8)
    position_tolerance = 1e-3  
    orientation_tolerance = 1e-2  
    if position_error > position_tolerance: # or orientation_error > orientation_tolerance:
        print("IK solution is not accurate enough.")
        #time.sleep(0.5)
        #input("IK solution is not accurate enough. Press enter to continue...")
    # ********************************************************************************************************************************************************************************************************

    # Once we have our joint positions from inverse kinematics, we simply repeat the process above to control the robot to those joints.
    current_joint_positions = robot.get_arm_joint_values() # Get the joint values of the robot in simulation currently
    joint_positions = np.array(ikJoints) # Set what we want the target robot joint values to be
    
    robot.control_arm_joints(joint_positions)
    p.stepSimulation(physicsClientId=client_id)

    #interpolate_positions = np.linspace(current_joint_positions, joint_positions, 5) # Create 1000 intermediate joint values to move the robot smoothly
    #for i in interpolate_positions: 
    #    robot.control_arm_joints(i) # Loop over all the intermediate positions and control the robot to that position
    #    p.stepSimulation(physicsClientId=client_id) # Since this simulation is time-based, we need to take a step forward in time to make sure the simulation actually updates
    #    time.sleep(1./240.)
    
def main():

    #%% ------------------------------- Robot Loading and Vis. -------------------------------
    global client_id, robot, robot_id, startup

    # Create a simulation client/GUI, then set the proper gravity settings
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    # Load robot into simulation
    robot_id, urdf_path = load_ur_robot(robot_initial_pose=[[0, 0, 0], [0, 0, 0, 1]], client_id=client_id, urdf_dir = "robots/urdf", urdf_file = "ur5e_fixed.urdf")
    robot = UR5Controller(robot_id, rng=None, client_id=client_id)

    print("The robot has loaded! If you open the simulation window, you should see the robot in the window. You may see a few small cubes--that's expected, those are some placeholder links we put there for future use.\n")
    input("Press ENTER to being tracking TouchX position updates.")

    #%% ------------------------------- Read TouchX updates -------------------------------
    device = HapticDevice(callback=device_callback, scheduler_type="async") # Initialize device and set callback
    global current_p_touchx, p_touchx_home

    try:
        print("Reading positions from Touch X\n")
        while True:
            # Read latest updated position from state
            x,y,z = device_state.position
            current_p_touchx = np.array([x, y, z])
            print(f"TX Position (mm): x={x:7.2f}, y={y:7.2f}, z={z:7.2f}", end="\r")

            # Store home position
            if startup:
                p_touchx_home = current_p_touchx
                print("Set home position")

            # Calculate new sim position
            sim_delta = findPositionDelta()
            new_p_sim = current_p_sim + sim_delta
            print(f"PB Position (mm): x={new_p_sim[0]:7.2f}, y={new_p_sim[1]:7.2f}, z={new_p_sim[2]:7.2f}", end="\r")

            # Move arm to new position
            current_p_sim = new_p_sim
            move_to_position(new_p_sim)
            if startup:
                input("Waiting... press ENTER to begin following TouchX position")
                startup = False

            time.sleep(1./240.)  # Thread runs at 1kHz but including delay
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Close device
        device.close()

if __name__ == "__main__":
    main()

p.disconnect(client_id)