import pybullet as p
import numpy as np
import pybullet_data
import pybullet_utils_cust as pu
import os
import time
from UR5Controller import UR5Controller
    
def load_ur_robot(robot_initial_pose=[[0., 0., 0.], [0., 0., 0., 1.]], 
                          client_id=0,urdf_dir="./robots/urdf/", urdf_file = "ur5e_fixed.urdf"):
    urdf_filepath = os.path.join(urdf_dir, urdf_file)  # Path to the URDF file, a file that contains information about the robot geometry and connections

    if not os.path.exists(urdf_filepath): # Check to make sure the path exists, and raise an error if it doesn't
        raise ValueError(f"URDF file {urdf_filepath} does not exist!")

    # Load the robot (by using the URDF file), and also assign it an inisial position and angle for the base (basePosition and baseOrientation)
    robot_id = p.loadURDF(urdf_filepath, 
                          basePosition=robot_initial_pose[0], 
                          baseOrientation=robot_initial_pose[1],
                          useFixedBase=True,
                          physicsClientId=client_id,
                          #flags=p.URDF_USE_SELF_COLLISION  # Enable self-collision checking
                        )
    return robot_id, urdf_filepath

#%% ------------------------------- Robot Loading and Vis. -------------------------------

# Create a simulation client/GUI, then set the proper gravity settings
client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81, physicsClientId=client_id)

# Use our function above to load the robot into the simulation
robot_id, urdf_path = load_ur_robot(robot_initial_pose=[[0, 0, 0], [0, 0, 0, 1]], client_id=client_id, urdf_dir = "robots/urdf", urdf_file = "ur5e_fixed.urdf")

# This is a custom controller we wrote (UR5Controller.py) that has wrappers for some functions that can be used to control the robot, both in simulation and physically.
robot = UR5Controller(robot_id, rng=None, client_id=client_id)

print("The robot has loaded! If you open the simulation window, you should see the robot in the window. You may see a few small cubes--that's expected, those are some placeholder links we put there for future use.\n")

#%% ------------------------------- Setting Joint Angles for the Robot -------------------------------

input("The next code will set the robot joint angles to a new position using the set_arm_joints function. Press enter, and you should see the robot immediately snap to a new pose...\n")


joint_positions = np.array([0, 0, 0, 0, 0, 0])
robot.set_arm_joints(joint_positions)

#%% ------------------------------- Smooth Control of Robot Joints -------------------------------

input("Usually, we want the robot to gradually move to a position, not immediately snap to a pose. So to do so, we usually interpolate between an initial pose and our target pose, then use control_arm_joints to slowly move the robot into position.\n")

current_joint_positions = robot.get_arm_joint_values() # Get the joint values of the robot in simulation currently
joint_positions = np.array([1.5, -1.5, 1.5, 1.5, 1.5, 1.5]) # Set what we want the target robot joint values to be in radians
interpolate_positions = np.linspace(current_joint_positions, joint_positions, 1000) # Create 1000 intermediate joint values to move the robot smoothly. You can change the number to make the robot move faster/slower.
for i in interpolate_positions: 
    robot.control_arm_joints(i) # Loop over all the intermediate positions and control the robot to that position
    p.stepSimulation(physicsClientId=client_id) # Since this simulation is time-based, we need to take a step forward in time to make sure the simulation actually updates
    time.sleep(1./240.)
    
#%% ------------------------------- Inverse Kinematics and Cartesian Commands -------------------------------
    
input("This is all nice, but a lot of times we don't want to give the robot commands using joint angles. We'd rather tell the robot to 'go to this XYZ coordinate', or face this 'Rx Ry Rz' direction. To do that, we would pass our target position/direction into Pybullet's inverse kinematic solver, which would spit out a joint angle for us to put back into the control_arm_joints function.\n")

target_position = np.array([0.4, 0.4, 0.4]) # Target XYZ position in meters
target_orientation = np.array([0, 0, 0, 1]) # Target rotation orientation, expressed as a quaternion in x, y, z, w format. You don't need to know what this means (though the textbook should explain it too), but feel free to mess around with this if you'd like to see

# This is the main function--this will take in some info about your target position + rotation, and spit out a set of joint angles you can give to the robot
ikJoints = p.calculateInverseKinematics(robot.id,
                                           8,
                                               target_position,
                                               target_orientation,
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
if position_error > position_tolerance or orientation_error > orientation_tolerance:
    input("IK solution is not accurate enough. Press enter to continue...")
# ********************************************************************************************************************************************************************************************************

# Once we have our joint positions from inverse kinematics, we simply repeat the process above to control the robot to those joints.
current_joint_positions = robot.get_arm_joint_values() # Get the joint values of the robot in simulation currently
joint_positions = np.array(ikJoints) # Set what we want the target robot joint values to be
interpolate_positions = np.linspace(current_joint_positions, joint_positions, 1000) # Create 1000 intermediate joint values to move the robot smoothly
for i in interpolate_positions: 
    robot.control_arm_joints(i) # Loop over all the intermediate positions and control the robot to that position
    p.stepSimulation(physicsClientId=client_id) # Since this simulation is time-based, we need to take a step forward in time to make sure the simulation actually updates
    time.sleep(1./240.)
    
# --------------------------------- END ------------------------------------
    
input("That's it! Feel free to play around with any of this code to get familiar with it--it's all in simulation so nothing's going to break regardless. With the functions above, you should be able to do something simple, like get the robot to move in a straight line, or do the initial writing thing I mentioned on Tuesday. Not mandatory, but could be good to get familiar with the library! Press enter to close the simulation.")

p.disconnect(client_id)