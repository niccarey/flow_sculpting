# Variables: enable hardware can be on or off

import pyatk
import pyaraas
from pyatk import Transform, Vector, Quaternion, Part, Actor, Gripper, Articulation, PlanningError
from pyaraas.tools import PathPlanner 

from admittance_controller import AdmittanceController

import gym

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import time
import numpy as np
import os, yaml,argparse
import math
import zmq
import copy

import json

import traceback
import logging 

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from rl_games.common import env_configurations # check rl games env requirements
    from rl_games.torch_runner import Runner

from rl_games.algos_torch import model_builder

config_name = "./med_gear_insertion_config.yaml" # Configuration file for the runner network.

# task and workcell info loaded from config file
with open(config_name, 'r') as stream:
    config = yaml.safe_load(stream)

log_file = config['params']['task']['log_name'] # to share logging info across both ends. (this is a debugging hack and may not be thread safe!)
roadmap_location =  config['params']['task']['roadmap_location']
WORKCELL_NAME= config['params']['task']['workcell_name']
msg_address = config['params']['task']['socket_addr']

logging.basicConfig(filename=log_file,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# TODO: can we get rid of constant swapping between pyatk and numpy transform reps? 

# =============================================================================
# Define some task constants, hard-coded workcell info, etc
enable_hardware = True

# This is a single robot control policy, so we reflect this in action unpacking, etc.

# WORKCELL CONSTANTS
# ----------------------------------

# Some useful workspace-specific poses 
WORLD_T_R0_HOME = Transform([436.46, 299.20, 533.14, np.pi, 0, 0])
WORLD_T_R1_HOME = Transform([-436.46, 299.20, 533.14, -np.pi, 0, 0])
R1_INTERIM = Transform(-406.34,560,603.81,-1.87518,1.10619,0)

# Define some safe joint poses to avoid IK-based windup:
SAFE_JOINT_0 = [-0.5460194357980299, -1.220403000524803, -2.145070509388365, -1.336707391811452, 1.556265651581378, 2.593192338943481]  
SAFE_JOINT_1 = [-1.873943290014628, -1.219899260009004, -2.138007362091444, -1.347187099941022, 1.56850971900005, -1.873934388160706]  


# TASK SPECIFIC CONSTANTS
# ----------------------------------- 
# ideally these can be constructed from digital twin part origins and grasp offsets, and perception information
# We can accommodate a few mm of imprecision in all of these.

# Robot positioninig transforms can be defined using world_to_flange or world_to_gripper TCP, 
# but need to keep these consistent - to avoid too many transform flips, we currently use world_to_flange poses
# to generate robot motion commands.

RQ85_TCP_OFFSET = Transform(0,0,285,0,0,0,1) # distance between robot wrist and gripper TCP point. 
# We can also pull this from araas - see define_offset_transform function. Hardcoded for transparency for now.

GEAR_TOOL_OFFSET = RQ85_TCP_OFFSET 

# define pickup and starting poses before starting trained insertion policy

# these constants are for the demo cell - peg location is the (world) transform of the nub on the stewart platform base
# replace with kitting peg location, known grasp offset, etc, when available

# gear origin is at grasp origin, give or take - update this if we change models or grasp points
# not sure why the gear pick pose is how it is, but anyway
GEAR_GRASP_OFFSET = Transform(0, 0, 10,0, 0, 0)
GEAR_PICK_POSE = GEAR_GRASP_OFFSET.multiply(Transform(325.18,406.95,32.86,0.000000,1.000000,-0.000000,0.000000)) #.multiply(GEAR_GRASP_OFFSET) # gear-in-world pose

# apparently ok? Whoops, forgot to change orientation from elbow
PEG_LOCATION =  Transform(275.18,290.95,32.86, 0.000000,1.000000,-0.000000,0.000000)
GEAR_PRE_INSERT = Transform(PEG_LOCATION)
GEAR_PRE_INSERT.position.z += 45

# OFFSETS FROM TRAINING CONDITIONS
# ------------------------------------------
# the assumption is that the workcell setup in isaac is the same as the workcell setup in real
# but sometimes origin poses, etc, may change (esp with dynamic workcell). 
# Input to the network is the robot pose relative to its own base, assuming that the base->target transform is fixed
# For a top-down insertion, if we rotate the robot base pose estimation by the yaw delta from new target to old target base transform
# and subtract any cartesian offset also from the robot base pose
# then we should get the right inputs to the system.
#board_origin_T_peg_base = ([0.2172, 0.0617, 0.2647], [-0.7071068, 0, 0, 0.7071068]) 
#WORLD_T_HOLE_START = TASKBOARD_POSE 
board_origin_T_peg_base = Transform(217.2, 61.7, 264.7, -0.7071068, 0, 0, 0.7071068)
WORLD_T_HOLE_START = Transform(150, 315, -34, 0.7071068, 0, 0, 0.7071068)
#print("Training regime: robot to target transform")
#print(WORLD_T_HOLE_START.multiply(board_origin_T_peg_base))
R_T_TRAIN = WORLD_T_HOLE_START.multiply(board_origin_T_peg_base)

# Robot pose always starts at origin during training, so we only need relative poses.
W_T_Robot = Transform(836.48,609.36,27.08,0.000,0,-1.57080)# real robot base pose
PEG_BASE_LOCATION = Transform(275.18,290.95,32.86, 0.000000,0, 0.000000,1.000000)
R_T_REAL = (W_T_Robot.invert()).multiply(PEG_BASE_LOCATION)
#print("Real regime: robot to target transform") # This seems a bit weird
#print(R_T_REAL)

REAL_T_TRAIN = (R_T_REAL.invert()).multiply(R_T_TRAIN) # fixed delta transform to map real poses to training origin schema
#print("Fixed offset")
#print(REAL_T_TRAIN)


# GENERAL CONTROL CONSTRAINTS
# ------------------------------------

# heuristic thresholds, timesteps, gains, etc (these are usually applicable across multiple contexts)
translation_limit = 0.05 # (cap translation actions at 50mm)
rotation_limit = 1/180 * np.pi # only relevant if we engage rotation in the policy

unit_scale = 1000 # for legacy reasons, controller gains are tuned for metres, not mm - we need to scale inputs (outputs are already in mm)

dt = 1./125.0 # robot control time delta

# some thresholding / sensor clipping values
force_limit = 100 
torque_limit = 1
t_vel_limit = 20  # 
r_vel_limit = 1

# gain constants and virtual mass matrix for admittance control
M_DEFAULT_LIN = 20
M_DEFAULT_ANG = 20
KP_DEFAULT = 200
KD_DEFAULT_LIN = 2 * np.sqrt(np.multiply(M_DEFAULT_LIN, KP_DEFAULT))
KD_DEFAULT_ANG = 2 * np.sqrt(np.multiply(M_DEFAULT_ANG, KP_DEFAULT))

# =============================================================================

# some useful classes:

class ForceSensorWrapper():

    def __init__(self, robot, force_sensor, h_size):
        self.h = h_size
        self.sensor = force_sensor
        self.robot = robot
        self.force = Vector(0,0,0)
        self.torque = Vector(0,0,0)
        
        # is there a reason to use numpy arrays here? just for ease of indexing?
        self.world_wrench = np.zeros(6) 
        self.wrench_bias = np.zeros(6)
        self.history_buffer = np.zeros([h_size, 6])# ditto

        self.sample = {}
        self.sample["force"] = self.force
        self.sample["moment"] = self.torque

    def update_force(self):
        # get a new sample, use windowing average to denoise, update latest reading + history buffer
        if enable_hardware:
            try:
                self.sample = self.sensor.sample()
                self.force = self.sample["force"]
                self.torque = self.sample["moment"]
            except:
                print("Reading from sensor failed")
                self.force = Vector(0,0,0)
                self.torque = Vector(0,0,0)

        else:
            # dummy read
            self.force = Vector(0,0,0)
            self.torque = Vector(0,0,0)

        force_bias = Vector(self.wrench_bias[:3])
        torque_bias = Vector(self.wrench_bias[3:])

        new_world_force = (self.robot.get_flange_transform().rotation.multiply(self.force)).subtract(force_bias)
        new_world_torque = (self.robot.get_flange_transform().rotation.multiply(self.torque)).subtract(torque_bias)

        # shuffle history
        self.history_buffer[:self.h-1,:] = self.history_buffer[1:self.h,:]
        self.history_buffer[-1,:3] = np.array([new_world_force.x, new_world_force.y, new_world_force.z])
        self.history_buffer[-1,3:] = np.array([new_world_torque.x, new_world_torque.y, new_world_torque.z])
        
        # average within the sliding window to de-noise the FT reading
        self.world_wrench[:3] = np.mean(self.history_buffer[:, :3], axis=0)
        self.world_wrench[3:] = np.mean(self.history_buffer[:, 3:], axis=0)


    def calculate_sensor_bias(self):
        buffer_length = self.h
        calibration_sample = 0

        while calibration_sample < buffer_length:
            # if we are running in sim: use dummy variables (could also hook into PhysX or Bullet)
            if enable_hardware:
                try:
                    self.sample = self.sensor.sample()
                    self.force = self.sample["force"]
                    self.torque = self.sample["moment"]
                except:
                    print("Reading from sensor failed")
                    self.force = Vector(0,0,0)
                    self.torque = Vector(0,0,0)

            else:
                # dummy read
                self.force = Vector(0,0,0)
                self.torque = Vector(0,0,0)

            new_world_force = self.robot.get_flange_transform().rotation.multiply(self.force)
            new_world_torque = self.robot.get_flange_transform().rotation.multiply(self.torque)

            self.history_buffer[calibration_sample, :3] = [new_world_force.x, new_world_force.y, new_world_force.z]
            self.history_buffer[calibration_sample, 3:] = [new_world_torque.x, new_world_torque.y, new_world_torque.z]

            calibration_sample += 1
        
        # use buffer mean as static bias estimate:
        self.wrench_bias[:3] = np.mean(self.history_buffer[:, :3], axis=0)
        self.wrench_bias[3:] = np.mean(self.history_buffer[:, 3:], axis=0)

        self.history_buffer = self.history_buffer - self.wrench_bias # update to account for bias
        self.world_wrench = self.history_buffer[self.h-1] 


    def force_sample_to_str(self): 
        # do we ever need this?
        # returns latest force sample as a json struct 
        return json.dumps({"force": self.sample["force"].get_values(), "moment": self.sample["moment"].get_values()})


class SampleBuffer():
    dt : float

    def __init__(self,init_sample, dt):
        # this structure only works for strict pose, we're gonna fall down with rotation if we use transforms with velocity.
        self.prev_reading = Transform(0,0,0,0,0,0)
        self.current_reading = Transform(init_sample)
        self.dt = dt

    # if we don't push new samples when using delta or diff, prev_reading is never actually used. Can simply take it out.

    def update(self,new_sample):
        # push latest reading into prev_reading, update latest_reading with new sample
        self.prev_reading = Transform(self.current_reading)
        self.current_reading = Transform(new_sample)

    def get_delta(self, reading):
        # return change between current and previous reading (as transform or numpy array?)
        pos_error = reading.position.subtract(self.current_reading.position)
        rot_error = (reading.rotation.multiply(self.current_reading.rotation.invert())).get_rx_ry_rz()

        return np.array([pos_error.x, pos_error.y, pos_error.z, rot_error.x, rot_error.y, rot_error.z])

    def differential(self, reading):
        # differentiate current and previous reading with timestep dt, expressed as (np.array?)
        delta = self.get_delta(reading)
        return delta/self.dt


class LinearBuffer():
    # minimal storage class for array-like objects
    def __init__(self, init_sample):
        self.prev_reading = np.zeros(init_sample.shape)
        self.current_reading = copy.deepcopy(init_sample)

    def update(self, new_sample):
        self.prev_reading = copy.deepcopy(self.current_reading)
        self.current_reading = copy.deepcopy(new_sample)


# some useful functions:

async def joint_path_execute(workcell, robot, goal_joints, duration):
    # It's not safe to just run a trajectory, so we have to check the plan first.
    planner = PathPlanner(workcell, 4, roadmap_location, False)
    await planner.plan_and_execute_raw_joint_trajectory(robot, goal_joints, duration)
    planner.shutdown()

async def araas_path_planner(workcell, robot, goal, duration):
    planner = PathPlanner(workcell, 4, roadmap_location, False)
    await planner.plan_and_execute_joint_trajectory(robot, goal, duration)
    planner.shutdown()

async def araas_small_move(robot, goal, duration):
    await robot.move_cartesian(goal, duration)

def define_offset_transform(robot, reference_actor):
    # check that the refernce actor is attached to the correct robot, then calculate the fixed flange-TCP offset

    w_T_flange = robot.get_flange_transform()
    w_T_tcp = reference_actor.get_tcp_transform()

    # check that the reference actor is attached to a robot:
    robot_parent = []
    try:
        robot_parent = reference_actor.get_attached_robot()
    except:
        raise PlanningError("Planning reference frame is not attached to a robot")

    if robot_parent is not robot:
        raise PlanningError("Planning reference frame is not attached to correct robot")

    # Calculate offset in flange frame:
    tcp_T_flange = pyatk.Transform((w_T_tcp.invert()).multiply(w_T_flange))

    return tcp_T_flange


# control- and communication-related functions

def bind_zmq_socket(ctx, address):
    # honestly, might be simpler to use REP/REQ
    s1 = ctx.socket(zmq.REP)
    s1.bind(address)
    return s1

def extract_action(robot, action):
    # action here can be assumed to be a 3-vector describing a translation motion
    # so we just threshold it and add a null rotation command. Other processes might require active rotation control.
    default_action = _compute_default_action(robot) # hmm why is this poor.
    # oh dang, need to either transform p-delta before scaling and then add, or de-normalize default action

    p_delta = unit_scale*translation_limit*np.asarray(action)
    r_delta = Quaternion(0,0,0,1) # null rotation if not controlling rotation

    # action applies to flange relative to current pose, assuming coordinates are in training world base. 
    # So: goal comes in as (action relative to robot base)
    # Do we need to take into account the offset transform between real target and training target, or real robot
    # base frame vs training robot base frame?
    # Feels like we do, because action comes in expressed in world coordinates. 
    # To convert back: create transform from delta:
    train_action = Transform(p_delta[0], p_delta[1], p_delta[2], r_delta.x, r_delta.y, r_delta.z, r_delta.w)

    real_action_position = (REAL_T_TRAIN.invert()).rotation.multiply(train_action.position)
    real_action_rotation = (REAL_T_TRAIN.invert()).rotation.multiply(train_action.rotation)

    default_action_position = Vector(default_action[:3])
    default_action_position.scale(unit_scale*translation_limit)

    default_action_rotation = Quaternion() #(Vector(default_action[3:])) #could also use set_rx etc EH not even using it anyway

    # apply weighting
    default_action_position.scale(0.5)
    real_action_position.scale(0.5)

    action_input_pos = default_action_position.add(real_action_position)
    # Rotations are NOT enabled, so just ignore all of that for now.

    p_start = robot.get_flange_transform().position
    p_goal = Vector(p_start.x + action_input_pos.x, p_start.y + action_input_pos.y, p_start.z+action_input_pos.z)
    r_goal = Quaternion(0,0,0,1)#robot.get_flange_transform().rotation.multiply(real_action_rotation)

    goal_pose = Transform(p_goal.x, p_goal.y, p_goal.z,r_goal.x, r_goal.y, r_goal.z, r_goal.w)
    logging.info("Goal pose calculated")

    return goal_pose

def _compute_default_action(robot):
    # Compute the current pose of the robot flange
    current_pose = robot.get_flange_transform()
    # oh wait need this to be nominal flange goal, duh
    GEAR_TCP = GEAR_GRASP_OFFSET.multiply(GEAR_TOOL_OFFSET.invert()) 
    nominal_goal_pose = PEG_LOCATION.multiply(GEAR_TCP)

    p_delta = nominal_goal_pose.position.subtract(current_pose.position)
    r_delta = (current_pose.rotation.invert()).multiply(nominal_goal_pose.rotation)
    # Convert to axis-angle representation for action calc
    r_delt_euler = Vector(r_delta.get_rx_ry_rz())

    # Combine position and rotation differences - converting to numpy representation for normalization purposes
    default_action = np.array([p_delta.x, p_delta.y, p_delta.z, r_delt_euler.x, r_delt_euler.y, r_delt_euler.z])
        
    # Normalize the action 
    default_action = default_action / np.linalg.norm(default_action + 1e-8)
        
    return default_action


def get_control_update(robot, 
        ft_wrapper, 
        controller,
        goal_pose, 
        pose_buffer):

    # TO DO: add velocity reading from driver (or velocity calculation at wrapper level instead of python level!)
    pos_error = robot.get_flange_transform().position.subtract(goal_pose.position)
    rot_error = (robot.get_flange_transform().rotation.multiply(goal_pose.rotation.invert())).get_rx_ry_rz()

    pose_error = np.array([pos_error.x, pos_error.y, pos_error.z, rot_error.x, rot_error.y, rot_error.z])

    vel_est = pose_buffer.differential(robot.get_flange_transform()) # when we add velocity sampling, can get this directly

    # desired velocity is always zero:
    vel_error = vel_est - np.zeros(6)

    ft_wrapper.update_force() # can we assume force has recently been initialised? if so, can probably take this out

    # check limits and clip the input sample if necessary:
    world_wrench = np.array(ft_wrapper.world_wrench)
    world_wrench_clip = controller.clip_input_wrench(world_wrench)

    # calculate a velocity command according to admittance control policy
    scaled_pose_error = (1/unit_scale)*pose_error 
    scaled_vel_error = (1/unit_scale)*vel_error
    linear_vel_cmd, rot_vel_cmd = controller.get_vel_cmd(world_wrench_clip, scaled_pose_error, scaled_vel_error, dt)

    # convert to robot flange pose and return:    
    robot_linear_vel = (pose_buffer.current_reading.rotation).invert().multiply(linear_vel_cmd)
    robot_rot_vel = ((pose_buffer.current_reading.rotation).invert().multiply(rot_vel_cmd))

    return robot_linear_vel, robot_rot_vel

def initialise_policy_states(robot0, socket):
    action_msg = socket.recv_json()

    if 'update' in action_msg:
        logging.info("received update state request")

        r0_ee_pose = robot0.get_flange_transform()

        # policy was trained in m (also! Check what the pose is relative to!?) must have a goal position offset, surely?
        r0_trans = 0.001*np.array([r0_ee_pose.position.x, r0_ee_pose.position.y, r0_ee_pose.position.z])
        r0_rot = np.array(r0_ee_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r0_state_return = {}
        vel_state_return = {}

        r0_state_return["trans"] = (r0_trans).tolist()
        r0_state_return["rot_c1"] = (np.array(r0_rot[:3, 0])).tolist()
        r0_state_return["rot_c2"] = (np.array(r0_rot[:3, 1])).tolist()
        
        state_message["robot_pose"] = r0_state_return

        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)

        # hopefully linear, angular velocity are in m already

        vel_state_return["linear vel"] = lin_vel.tolist()
        vel_state_return["angular vel"] = ang_vel.tolist()
        state_message["robot_vel"] = vel_state_return

        socket.send_json(state_message) 


    else:
        print("Unexpected message recieved from socket")
        print(action_msg)
        logging.info("Initialization error")


async def insertion_command_distributor(robot, ft, action_time, socket):

    logging.info("launching robot control updater")
    # set up buffer structures for forces, velocities, etc (velocity currently unused)

    # initialise buffers:
    r_flange_velocity = np.zeros(6)
    r_pose_init = robot.get_flange_transform()

    r_pose_buffer = SampleBuffer(r_pose_init, dt)

    # we actually don't use this, AND it shouldn't be a sample buffer, which expects proper pose input
    # can use it for observation updates I guess
    r_vel_buffer = LinearBuffer(r_flange_velocity)

    # initialise force sensor data wrappers
    history_size = 6 # we use a history buffer as a moving window to smooth force sensor readings - this could also be built into araas
    f_data = ForceSensorWrapper(robot, ft, history_size)

    # when using OnRobot sensors, as long is the load is light we can sample load-compensated readings and eliminate the bias zeroing
    # (this also allows better control of rotational dimensions)

    f_data.calculate_sensor_bias()

    # Initialise controllers 
    mass_matrix = np.array([M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_ANG, M_DEFAULT_ANG, M_DEFAULT_ANG])
    kp_matrix = np.array([KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kd_matrix = np.array([KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_ANG, KD_DEFAULT_ANG, KD_DEFAULT_ANG])

    r_admit_control = AdmittanceController(mass_matrix, kp_matrix, kd_matrix, force_limit, torque_limit, 1000, t_vel_limit, r_vel_limit)

    done = False
    logging.info("Action distributor initialized, waiting for action command")
    
    while not done:
        # wait for action state message
        action_msg = socket.recv_json()

        # check for terminal condition 
        if 'done' in action_msg:
            print("Received termination condition")
            logging.info("Insertion finished")
            done = True
            continue

        # unpack into robot velocity commands
        # For this insertion configuration, it's only 3x cartesian dimensions per robot
        logging.info("received action command, unpacking")
        robot_action = action_msg["robot_action"]

        robot_goal = extract_action(robot, robot_action) # <-- also adds a default component 

        #  add the following lines for delta and control command

        init_time = time.time()
    
        while((time.time() - init_time) < action_time): 
            #if we go too fast, we don't have time to converge on the goal speed
            r_lin_cmd, r_rot_cmd = get_control_update(robot, f_data,r_admit_control, robot_goal,r_pose_buffer)

            # send commands to hardware
            _, r_pose_update = robot.set_cartesian_velocity(r_lin_cmd, r_rot_cmd, dt, force_limit, mass=1)

            # not entirely sure why control delay/state update is structured like this, could be cleaner
            time.sleep(dt/2) 

            r_pose_buffer.update(r_pose_update)

            # update velocity, though it's not really used just now (vel is approximated from pose delta when we need it)
            new_vel = r_pose_buffer.differential(r_pose_update)

            r_vel_buffer.update(new_vel)
            f_data.update_force()


        # construct observation message:
        logging.info("Finished control action")

        real_flange_pose = robot.get_flange_transform()

        # Updates to observation - what do we send back to the network?
        # Want to send back the pose of the flange relative to the robot, adjusted for training environment
        # so, get flange-in-robot, then transform real-to-training
        robot_base_pose = robot.get_base_transform() 
        flange_in_robot = (robot_base_pose.invert()).multiply(real_flange_pose)

        # transform to match training environment
        obs_flange = REAL_T_TRAIN.multiply(flange_in_robot)    


        r_trans = 0.001*np.array([obs_flange.position.x, obs_flange.position.y, obs_flange.position.z])
        r_rot = np.array(obs_flange.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r_state_return = {}
        vel_state_return = {}

        r_state_return["trans"] = (r_trans).tolist()
        r_state_return["rot_c1"] = (np.array(r_rot[:3, 0])).tolist()
        r_state_return["rot_c2"] = (np.array(r_rot[:3, 1])).tolist()

        vel_est = r_vel_buffer.current_reading
        vel_state_return["linear vel"] = (vel_est[:3]).tolist()
        vel_state_return["angular vel"] = (vel_est[3:]).tolist()
        
        state_message["robot_pose"] = r_state_return
        state_message["robot_vel"] = vel_state_return

        socket.send_json(state_message) 

        logging.info("Observation sent, preparing to receive")

    print("insertion task complete")




# =================================================================
# main script!


# initialise araas
if(__name__=="__main__"):

    # initialise workcell
    workcell = pyaraas.start(WORKCELL_NAME, enable_hardware=enable_hardware)

    # get workcell, robot, gripper objects:
    # We initialise UR10e-1 just to ensure it's out of the way
    null_robot = workcell.get_robot("UR10e-1") #check these 
    gear_robot = workcell.get_robot("UR10e-0")
    null_gripper = workcell.get_gripper("Rq85-1")
    gear_gripper = workcell.get_gripper("Rq85-0")

    # get force sensors
    gear_f_sensor = workcell.get_force_sensor('UR10e-0')

    # go to workcell default start poses
    print("Joint driving to safe poses to eliminate windup") # this plans joint-to-joint motion, while checking for collisions

    # why isn't try/catch picking this up?
    try:
        #if not (gear_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)) and not (null_robot.get_flange_transform().is_equal(0.1, 0.01, WORLD_T_R0_HOME)):
        #    pyaraas.run([pyaraas.Task(joint_path_execute, workcell, gear_robot, SAFE_JOINT_1, 5.0), pyaraas.Task(joint_path_execute, workcell, null_robot, SAFE_JOINT_0, 5.0)])        
        if not (gear_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)):
            pyaraas.run(pyaraas.Task(joint_path_execute, workcell, gear_robot, SAFE_JOINT_1, 5.0))
        #elif not (null_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R0_HOME)):
        #        pyaraas.run(pyaraas.Task(joint_path_execute, workcell, null_robot, SAFE_JOINT_0, 5.0))

    except Exception:
        traceback.print_exc()
        
    # do grasping and pickup
    # create an offset transform for picking items off the kit:
    approach = Transform([0, 0, 100, 0, 0, 0])

    # (should not need to actively set collision models right after initialisation, but can do so just as a failsafe)
    null_robot.set_collision_model(True, True)
    null_gripper.set_collision_model(True, True)
    gear_robot.set_collision_model(True, True)
    gear_gripper.set_collision_model(True, True)

    # pick up gear:
    gear_grasp_pose = GEAR_PICK_POSE.multiply(GEAR_TOOL_OFFSET.invert())
    gear_pregrasp_pose = approach.multiply(gear_grasp_pose)

    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, gear_robot, gear_pregrasp_pose, 5.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit()

    print("Picking up")

    # Collision disabling code block, to avoid spurious kit/table/taskboard collisions
    # ----------------------------
    gear_robot.set_collision_model(False, False)
    gear_gripper.set_collision_model(False, False)
    # ----------------------------

    # move down to grasp point(s), perform grasp(s), move back up to pre-grasp pose
    # 'small move' performs a linear cartesian motion in robot flange space

    pyaraas.run(pyaraas.Task(araas_small_move, gear_robot, gear_grasp_pose, 1.0))
    pyaraas.run(pyaraas.Task(gear_gripper.open, 0, 1))
    pyaraas.run(pyaraas.Task(araas_small_move, gear_robot, gear_pregrasp_pose, 1.0))

    print("Completed gear pickup")

    print("Moving to pre-insertion position")

    # enable collisions for large motions - also, ensure stewart platform concavity
    # -----------------------------
    gear_robot.set_collision_model(True, True)
    gear_gripper.set_collision_model(True, True)
    # nowhere near stand: but make sure we have the other cell features enabled
    #stand = workcell.get_peripheral("stand_for_stewart_platform_assembly_no_motor-0")
    #stand.set_collision_model(True, True)
    # -----------------------------

    # performing pre-insertion motions one at a time to avoid collision mid-motion (as planner has static workcell state)
    # correct for grasp orientation, calculate flange position
    GEAR_TCP = GEAR_GRASP_OFFSET.multiply(GEAR_TOOL_OFFSET.invert()) 
    GEAR_START_INSERT = GEAR_PRE_INSERT.multiply(GEAR_TCP) # starting flange transform 
    
    buffer_pose = Transform(GEAR_START_INSERT)
    buffer_pose.position.z += 200
    buffer_pose.position.y -= 0 #100
            
    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, gear_robot, buffer_pose, 3.0))
        pyaraas.run(pyaraas.Task(araas_small_move, gear_robot, GEAR_START_INSERT, 2.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 

    # Collision disabling code block, necessary before running an insertion policy
    # ----------------------------
    gear_robot.set_collision_model(False, False)
    gear_gripper.set_collision_model(False, False)
    #stand.set_collision_model(False, False)
    # ----------------------------

    print("Finished moving to task start configuration")
    # initialise communication socket to talk to policy:
    ctx = zmq.Context()
    socket = bind_zmq_socket(ctx, msg_address) 

    # action time is heuristic value mostly dependent on real world hardware
    # we found 0.4s was a good value too long and we get drift, too short and behaviour is jerky
    max_action_time = 0.5

    # check socket for initialising state request / reset request, initialise policy runner:
    initialise_policy_states(gear_robot, socket)

    # command distributor waits for query/action input from policy runner, enacts action, responds with new state
    try:
        pyaraas.run(pyaraas.Task(insertion_command_distributor, gear_robot, gear_f_sensor, max_action_time, socket))
    except Exception:
        traceback.print_exc()
        import sys
        sys.exit()

    # when we have returned: terminate socket and context
    socket.close()
    ctx.term()

    # Do grasp release and cleanup (TODO add this)
    pyaraas.run(pyaraas.Task(gear_gripper.open, 50, 1.0))
    pyaraas.run(pyaraas.Task(araas_small_move, gear_robot, gear_pregrasp_pose, 1.0))


