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

# This is a tricky one, because we need to pick up the bolt as well, probably also using an insertion policy.
# Try: elbow-peg insertion with appropriate workcell transform mods, then
# bolt-eye insertion.

# TODO: reconfigure learned robot action space to be relative to something more sensible (it used to be relative to estimated goal position)

# Process:
# - have UR10e-1 hold STRUT in position (we assume that this is post articulation reconfig)
# - UR10e-0 go to bolt pick, do bolt pickup action using elbow peg insertion with appropriate deltas
# - UR10e-0 go to bolt insert, do bolt insert action using bolt eye insertion with appropriate deltas

# code building: do bolt pick and insert placement first, then practice calling runners (need to trigger two runners sequentially! Time for the inference server?)
# then add strut hold mechanics.
# SLIGHTLY inefficient to switch holding robots, but not sure UR10-e0 has the reach even if we do the transform appropriately.

config_name = "./bolt_eye_insertion_config.yaml" # Configuration file - 
# we need to make sure each runner has the right network, and we call the right runner at the right time.
# for here it's just the workcell name that matters really.
# Update: cannot use the same socket in quick succession. Could either change sockets, or just wait a moment?
# Maybe closing it better will help

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
enable_hardware = False

# This is a single robot control policy, so we reflect this in action unpacking, etc.

# WORKCELL CONSTANTS
# ----------------------------------

# Some useful workspace-specific poses 
WORLD_T_R0_HOME = Transform([436.46, 299.20, 533.14, np.pi, 0, 0])
WORLD_T_R1_HOME = Transform([-470.46, 402.86, 604.61, -np.pi, 0, -np.pi/2])


R1_INTERIM = Transform(-406.34,560,603.81,-1.87518,1.10619,0)

# Now using R0 (would we have to retrain for R1? Observation is in workcell frame, so maybe yeah)
# workcell/robot frame has advantages and disadvantages - if adding force to the pipeline, we want a representation of global geometry
# however it makes it harder to adjust for new workcell layouts - need to add compensation transforms if we change anything.
# ... I think the latter is a problem, tbh - we would have to retrain on every new layout or stack. If 
# state position observation is (gripper TCP->estimated target), it will be much easier to adjust.

# Define some safe joint poses to avoid IK-based windup:
#SAFE_JOINT_0 = [-0.5460194357980299, -1.220403000524803, -2.145070509388365, -1.336707391811452, 1.556265651581378, 2.593192338943481]  
#SAFE_JOINT_1 = [-1.873943290014628, -1.219899260009004, -2.138007362091444, -1.347187099941022, 1.56850971900005, -1.873934388160706]  

SAFE_JOINT_0 = [-0.546019, -1.22040, -2.14507, -1.33670, 1.55627, 2.593192]  
SAFE_JOINT_1 = [0.62744587, -2.11185, 2.11185, -1.5707963267948966, -1.5707963267948966, 2.19422794]

# TASK SPECIFIC CONSTANTS
# ----------------------------------- 
# ideally these can be constructed from digital twin part origins and grasp offsets, and perception information
# We can accommodate a few mm of imprecision in all of these.

# Robot positioninig transforms can be defined using world_to_flange or world_to_gripper TCP, 
# but need to keep these consistent - to avoid too many transform flips, we currently use world_to_flange poses
# to generate robot motion commands.


RQ85_TCP_OFFSET = Transform(0,0,302,0,0,0,1) # distance between robot wrist and gripper TCP point. 
RATCHET_TCP_OFFSET = Transform(0,0,295,0,0,0,1)

# We can also pull this from araas - see define_offset_transform function. Hardcoded for transparency for now.

STRUT_TOOL_OFFSET = RQ85_TCP_OFFSET 
BOLT_TOOL_OFFSET = RATCHET_TCP_OFFSET

# define pickup and starting poses before starting trained insertion policy

# these constants are for the demo cell - peg location is the (world) transform of the nub on the stewart platform base
# replace with kitting peg location, known grasp offset, etc, when available

BOLT_WORLD_POSE = Transform(368,825,35,np.pi,0,0)
BOLT_GRASP_OFFSET = Transform(0,0,-17,0,0,0) # where we want to be when we start doing the pickup, I think
BOLT_PICKUP_START = BOLT_WORLD_POSE.multiply(BOLT_GRASP_OFFSET)

# Check strut origin and grasp pose in real
STRUT_LEAN_POSE = Transform(75.00,680.00,240.00,-0.34907,-0.08727,1.57080)
#Transform(-80.77999877929688, 195.26998901367188, 0, 0, 0, 0) #-1.5707963705062866, -1.5707963705062866, 0.0)
STRUT_GRASP_OFFSET = Transform(10, 0, 80.0, 0.5, -0.5, -0.5, 0.5)
STRUT_HOLD_POSE = STRUT_LEAN_POSE.multiply(STRUT_GRASP_OFFSET) # world pose of gripper when picking up elbow, including grasp offset

# ============================================================================
# Useful transforms 
# ============================================================================
# WORKCELL TRANSFORMS
PLATFORM_POSE = Transform(0.00, 809, 6, 0, 0, -0.9659258, 0.258819) # mounting offset - slightly above table, rotated 120 degrees around Z
PLATFORM_T_ELBOW = Transform(0.00, 150, 168, 0, 0, 0.70711, 0.70711) 
PEG_LOCATION = Transform(80, 685, 238, 0.0, 0.0, -1.047197699546814) 
BOLT_INSERT_START =Transform(49,440.57,510,-1.83260,0.00000,np.pi) #Starting point for flange

# should be flange transform when bolt is inserted at estimated goal pose.
W_T_FLANGE_GOAL = Transform(49, 451,503, -1.83260,0.00000,np.pi)


# TRAINING TRANSFORMSW

#((0.16649022698402405, 0.4282906949520111, 0.5167390704154968), (-0.20533481240272522, 0.766319990158081, 0.5880191326141357, -0.15755923092365265))
WORLD_T_TRAIN_GOAL = Transform(166.4, 428.3, 516.7, -0.20533481240272522, 0.766319990158081, 0.5880191326141357, -0.15755923092365265)
# There's a 90 degree offset in Z, and it's making it difficult to manually assess the relative transforms. Can get rid of it just for the sake of sanity
# (I think it probably fine though)

# This looks wrong, actually

# MOTION PLANNING TRANSFORMS
R1_ELBOW_PREGRASP = Transform(-165.50,512.59,396.39,-1.57076,0.97512,0.00000)
R1_ELBOW_PLANNED_POSE = Transform(R1_ELBOW_PREGRASP)
R1_ELBOW_PLANNED_POSE.position.y = 380

R1_ELBOW_GRASP = Transform(R1_ELBOW_PREGRASP)
R1_ELBOW_GRASP.position.z = PEG_LOCATION.position.z - 10 # approximate elbow height, minus grasp distance - may need to adjust

# =============================================================================
# Need to reconfigure the flange pose in real to flange pose in training space
# Equivalent to multiplying the flange pose by the transform between bolt world pose + grasp offset -> peg pose

# Peg location is WORLD_T_PEG
# Can just use BOLT_PICKUP_START, this corrects for a non longer extant problem
W_BOLT_START = BOLT_WORLD_POSE.multiply(BOLT_GRASP_OFFSET)
BOLT_T_PEG = W_BOLT_START.invert().multiply(PEG_LOCATION)
pickup_calib_transform = Transform(BOLT_T_PEG)

# Should be transforming from the araas frame to the world frame. Something is up with the bolt goals in isaac output, there's a 180 flip?
# ah ok. We send flange info back to the network, so we should construct this from flange tansforms.

# 1) figure out world_T_flangegoal in training 
# then same in execution, then construct the inversion.

INSERT_T_TRAIN = (W_T_FLANGE_GOAL.invert()).multiply(WORLD_T_TRAIN_GOAL)
insert_calib_transform = Transform(INSERT_T_TRAIN)
#print("Transform between training and real world")
#print(insert_calib_transform)

# Still seems wrong. Let's do a reverse check:
sanity_check = W_T_FLANGE_GOAL.multiply(insert_calib_transform)
print(sanity_check.get_values())

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
# could stand to be a bit stiffer, tbh
M_DEFAULT_LIN = 20
M_DEFAULT_ANG = 20
KP_DEFAULT = 400
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

def extract_action(robot, action, nominal_goal_pose):
    # action here can be assumed to be a 3-vector describing a translation motion
    # so we just threshold it and add a null rotation command. Other processes might require active rotation control.

    # Action is expressed in world frame and should not need adjusting as long as training regime
    # and execution environment match

    # Can add a default action computation to more closely mimic action results in training
    default_action = _compute_default_action(robot, nominal_goal_pose)

    default_action_position = Vector(default_action[:3])
    default_action_position.scale(unit_scale*translation_limit)

    default_action_rotation = Quaternion() #(Vector(default_action[3:])) #could also use set_rx etc EH not even using it anyway

    # apply weighting
    default_action_position.scale(0.5)

    p_delta = unit_scale*translation_limit*np.asarray(action)
    r_delta = Quaternion(0,0,0,1) # null rotation if not controlling rotation

    trained_action = Transform(p_delta[0], p_delta[1], p_delta[2], r_delta.x, r_delta.y, r_delta.z, r_delta.w)
    trained_action_position = trained_action.position 
    trained_action_position.scale(0.5)

    action_input_pos = default_action_position.add(trained_action_position)
    # Rotations are NOT enabled, so completely ignoring all of that for now.

    p_start = robot.get_flange_transform().position
    p_goal = Vector(p_start.x + action_input_pos.x, p_start.y + action_input_pos.y, p_start.z+action_input_pos.z)
    r_goal = robot.get_flange_transform().rotation.multiply(r_delta) # essentially null rotation - ah, hmm.
    # if the goal rotation is applied to the goal pose transform, then we got some issues.
    # Goal pose position looks right, so there must be an inversion applied when distributing to the controller

    goal_pose = Transform(p_goal.x, p_goal.y, p_goal.z,r_goal.x, r_goal.y, r_goal.z, r_goal.w)
    logging.info("Goal pose calculated")

    return goal_pose

def _compute_default_action(robot, nominal_goal_pose):
    # Compute the current pose of the robot flange
    current_pose = robot.get_flange_transform()

    # oh wait need this to be nominal flange goal, duh
    p_delta = nominal_goal_pose.position.subtract(current_pose.position)
    r_delta = (current_pose.rotation.invert()).multiply(nominal_goal_pose.rotation)

    # Convert to axis-angle representation for action calc
    r_delt_euler = Vector(r_delta.get_rx_ry_rz())

    # Combine position and rotation differences - converting to numpy representation for normalization purposes
    default_action = np.array([p_delta.x, p_delta.y, p_delta.z, r_delt_euler.x, r_delt_euler.y, r_delt_euler.z])
        
    # Normalize the action 
    default_action = default_action / np.linalg.norm(default_action + 1e-8)
    # weight Z? Probably not necessary, can't recall whether we did this during training ...
    #default_action[0] *= 0.7
    #default_action[1] *= 0.7
    #default_action[2] *= 1.35
        
    return default_action


def get_control_update(robot, 
        ft_wrapper, 
        controller,
        goal_pose, 
        pose_buffer):

    # TO DO: add velocity reading from driver (or velocity calculation at wrapper level instead of python level!)
    pos_error = robot.get_flange_transform().position.subtract(goal_pose.position)
    rot_error = (robot.get_flange_transform().rotation.multiply(goal_pose.rotation.invert())).get_rx_ry_rz() # this should be null

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

def initialise_pickup_states(robot0, socket):
    action_msg = socket.recv_json()

    if 'update' in action_msg:
        logging.info("received update state request")

        r0_ee_pose = robot0.get_flange_transform()
        # Policy was trained in m, need to convert
        obs_pose = pickup_calib_transform.multiply(r0_ee_pose)
        r0_trans = 0.001*np.array([obs_pose.position.x, obs_pose.position.y, obs_pose.position.z])
        r0_rot = np.array(obs_pose.extract_row_major()).reshape((4,4))

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

        vel_state_return["linear vel"] = lin_vel.tolist()
        vel_state_return["angular vel"] = ang_vel.tolist()
        state_message["robot_vel"] = vel_state_return

        socket.send_json(state_message) 

    else:
        print("Unexpected message recieved from socket")
        print(action_msg)
        logging.info("Initialization error")



def initialise_insert_states(robot0, socket):
    action_msg = socket.recv_json()

    if 'update' in action_msg:
        logging.info("received update state request")

        r0_ee_pose = robot0.get_flange_transform()
        # Policy was trained in m, need to convert
        obs_pose = r0_ee_pose.multiply(insert_calib_transform)
        r0_trans = 0.001*np.array([obs_pose.position.x, obs_pose.position.y, obs_pose.position.z])
        r0_rot = np.array(obs_pose.extract_row_major()).reshape((4,4))

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

        vel_state_return["linear vel"] = lin_vel.tolist()
        vel_state_return["angular vel"] = ang_vel.tolist()
        state_message["robot_vel"] = vel_state_return

        socket.send_json(state_message) 

    else:
        print("Unexpected message recieved from socket")
        print(action_msg)
        logging.info("Initialization error")


async def bolt_pickup_policy(robot, ft, action_time, socket):

    logging.info("launching robot control updater")
    # set up buffer structures for forces, velocities, etc (velocity currently unused)

    # initialise buffers:
    r_flange_velocity = np.zeros(6)
    r_pose_init = robot.get_flange_transform()

    r_pose_buffer = SampleBuffer(r_pose_init, dt)
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

    BOLT_TCP = BOLT_GRASP_OFFSET.multiply(BOLT_TOOL_OFFSET.invert()) 
    nominal_goal_pose = BOLT_WORLD_POSE.multiply(BOLT_TCP)  # this is REAL goal pose for robot flange, used to calculate default action

    while not done:
        # wait for action state message
        action_msg = socket.recv_json()

        # check for terminal condition 
        if 'done' in action_msg:
            print("Received termination condition")
            logging.info("Insertion finished")
            done = True            
            print("pickup task complete")
            return 

        # unpack into robot velocity commands
        # For this insertion configuration, it's only 3x cartesian dimensions per robot
        logging.info("received action command, unpacking")
        robot_action = action_msg["robot_action"]

        # Add a nominal pose we can use to calculate a default action
        robot_goal = extract_action(robot, robot_action, nominal_goal_pose)

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

        # Adjust the workcell observation to match the training state origin
        obs_pose = pickup_calib_transform.multiply(robot.get_flange_transform())
        r_trans = 0.001*np.array([obs_pose.position.x, obs_pose.position.y, obs_pose.position.z])
        r_rot = np.array(obs_pose.extract_row_major()).reshape((4,4))

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




async def insertion_command_distributor(robot, ft, action_time, socket):

    logging.info("launching robot control updater")
    # set up buffer structures for forces, velocities, etc (velocity currently unused)

    # initialise buffers:
    r_flange_velocity = np.zeros(6)
    r_pose_init = robot.get_flange_transform()

    r_pose_buffer = SampleBuffer(r_pose_init, dt)
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

    # Calculate nominal goal pose: Should be in flange coordinates
    BOLT_TCP = BOLT_GRASP_OFFSET.multiply(BOLT_TOOL_OFFSET.invert()) 
    nominal_goal_pose = W_T_FLANGE_GOAL.multiply(BOLT_TCP) 
    print("nominal goal: adjust bolt TCP to change")
    print(nominal_goal_pose.get_values())

    # network action output is clearly garbage so something is wrong with our observations
    while not done:
        # wait for action state message
        action_msg = socket.recv_json()

        # check for terminal condition 
        if 'done' in action_msg:
            print("Received termination condition")
            logging.info("Insertion finished")
            done = True
            print("insertion task complete")
            return 

        # unpack into robot velocity commands
        # For this insertion configuration, it's only 3x cartesian dimensions per robot
        logging.info("received action command, unpacking")
        robot_action = action_msg["robot_action"]

        # Add a nominal pose we can use to calculate a default action
        robot_goal = extract_action(robot, robot_action, nominal_goal_pose)

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

        # Adjust the workcell observation to match the training state origin
        obs_pose = (robot.get_flange_transform()).multiply(insert_calib_transform)
        r_trans = 0.001*np.array([obs_pose.position.x, obs_pose.position.y, obs_pose.position.z])
        r_rot = np.array(obs_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r_state_return = {}
        vel_state_return = {}
        
        r_state_return["trans"] = (r_trans).tolist()
        r_state_return["rot_c1"] = (np.array(r_rot[:3, 0])).tolist()
        r_state_return["rot_c2"] = (np.array(r_rot[:3, 1])).tolist()

        print("Sending state message as:")
        print(r_state_return["trans"])
        print(r_state_return["rot_c1"])

        vel_est = r_vel_buffer.current_reading
        vel_state_return["linear vel"] = (vel_est[:3]).tolist()
        vel_state_return["angular vel"] = (vel_est[3:]).tolist()

        state_message["robot_pose"] = r_state_return
        state_message["robot_vel"] = vel_state_return

        socket.send_json(state_message) 

        logging.info("Observation sent, preparing to receive")



# =================================================================
# main script!


# initialise araas
if(__name__=="__main__"):

    # initialise workcell
    workcell = pyaraas.start(WORKCELL_NAME, enable_hardware=enable_hardware)

    # get workcell, robot, gripper objects:
    # NO GRIPPER ON UR10e-0! Upgrade to refactored araas before attempting tool changing or ratchet driving, I think

    # Use UR0 to insert bolt, UR1 to hold strut
    bolt_robot = workcell.get_robot("UR10e-0")  
    bolt_gripper = workcell.get_peripheral("Ratchet_HexHead-0") #hope this works
    strut_robot = workcell.get_robot("UR10e-1")
    strut_gripper = workcell.get_gripper("Rq85-1")

    # get force sensor(s)
    bolt_f_sensor = workcell.get_force_sensor('OnRobotFT_HexQC-0')
    strut_f_sensor = workcell.get_force_sensor('OnRobotFT_HexQC-1')

    #init_sample = bolt_f_sensor.sample()
    # NOTHING SHOULD HAVE HAPPENED YET
    #print(init_sample["force"]) 
    #print(init_sample["moment"])
    
    # ------- DEBUG --------------
    # Visualize bolt pose:
    workcell.add_part("Precision-Shoulder-Screw-92981A814_18-8-Stainless-Steel", "bolt1", BOLT_WORLD_POSE)

    # Visualise strut pose (if we include this, need to turn collisions off right away)
    #workcell.add_part("strut", "strut1", STRUT_LEAN_POSE)

    # go to workcell default start poses
    print("Joint driving to safe poses to eliminate windup") # this plans joint-to-joint motion, while checking for collisions

    try:
        if not (bolt_robot.get_flange_transform().is_equal(5, 0.05, WORLD_T_R0_HOME)):
            pyaraas.run(pyaraas.Task(joint_path_execute, workcell, bolt_robot, SAFE_JOINT_0, 5.0))

        if not (strut_robot.get_flange_transform().is_equal(5, 0.05, WORLD_T_R1_HOME)):
            pyaraas.run(pyaraas.Task(joint_path_execute, workcell, strut_robot, SAFE_JOINT_1, 5.0))

    except Exception:
        traceback.print_exc()
        #import sys
        #sys.exit()
    
    # Go to prelim pickup pose

    # do grasping and pickup
    # create an offset transform for picking items off the kit:
    approach = Transform([0, 0, 160, 0, 0, 0])

    # (should not need to actively set collision models right after initialisation, but can do so just as a failsafe)
    bolt_robot.set_collision_model(True, True)
    bolt_gripper.set_collision_model(True, True)
    strut_robot.set_collision_model(True, True)
    strut_gripper.set_collision_model(True, True)


    print("Picking up bolt using elbow insertion runner")

    # Collision disabling code block, to avoid spurious kit/table/taskboard collisions
    # ----------------------------
    bolt_robot.set_collision_model(True, True)
    bolt_gripper.set_collision_model(True, True)
    stand = workcell.get_peripheral("stand_for_stewart_platform_assembly_no_motor-0")
    stand.set_collision_model(True, True)
    # ----------------------------

    # move down to grasp point(s), perform grasp(s), move back up to pre-grasp pose
    # 'small move' performs a linear cartesian motion in robot flange space

    bolt_grasp_pose = BOLT_PICKUP_START.multiply(BOLT_TOOL_OFFSET.invert())
    bolt_pregrasp_pose = approach.multiply(bolt_grasp_pose)

    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, bolt_robot, bolt_pregrasp_pose, 4.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit()

    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, bolt_grasp_pose, 1.0))

    # now, run insertion policy to get ratchet into bolt

    ctx = zmq.Context()
    socket = bind_zmq_socket(ctx, msg_address) 
    max_action_time = 0.4

    # check socket for initialising state request / reset request, initialise policy runner:
    initialise_pickup_states(bolt_robot, socket)

    try:
        pyaraas.run(pyaraas.Task(bolt_pickup_policy, bolt_robot, bolt_f_sensor, max_action_time, socket))
    except Exception:
        traceback.print_exc()
        #import sys
        #sys.exit()

    time.sleep(1.0)
    # need time for policy runner to exit
    socket.close()
    ctx.term()

    # Assuming success, continue:
    pickup_buffer = Transform(bolt_robot.get_flange_transform())
    pickup_buffer.position.z -= 2

    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, pickup_buffer, 1.0))

    pickup_buffer.position.z += 200
    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, pickup_buffer, 1.0))

    print("Completed bolt pickup")

    #pyaraas.run(pyaraas.Task(strut_gripper.open, 35,1))

    #strut_grasp_pose = STRUT_LEAN_POSE.multiply(STRUT_TOOL_OFFSET.invert())
    # R1_STRUT_PLANNED_POSE
    #try:
    #    pyaraas.run(pyaraas.Task(araas_path_planner, workcell, strut_robot, R1_STRUT_PLANNED_POSE, 3.0))
    #except Exception:
    #    print(traceback.format_exc())
    #    import sys
    #    sys.exit()


    #pyaraas.run(pyaraas.Task(strut_gripper.open, 40, 1))
    #pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, R1_ELBOW_PREGRASP, 1.0))
    #pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, R1_ELBOW_GRASP, 1.0))
    #pyaraas.run(pyaraas.Task(strut_gripper.open, 0, 1))

    print("Moving to pre-insertion position")

    # enable collisions for large motions - also, ensure stewart platform concavity
    # -----------------------------
    bolt_robot.set_collision_model(True, True)
    bolt_gripper.set_collision_model(True, True)
    strut_robot.set_collision_model(True, True)
    strut_gripper.set_collision_model(True, True)
    # -----------------------------

    # ... can we no longer input 6-dof transforms?
    bolt_pre_insert_pose = Transform(53,374.45,527.57,-1.83260,0.00000,1.56556)
            
    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, bolt_robot, bolt_pre_insert_pose, 6.0))
        pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, BOLT_INSERT_START, 2.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 


    # Collision disabling code block, necessary before running an insertion policy
    # ----------------------------
    bolt_robot.set_collision_model(False, False)
    bolt_robot.set_collision_model(False, False)
    strut_robot.set_collision_model(False, False)
    strut_gripper.set_collision_model(False, False)
    stand.set_collision_model(False, False)
    # ----------------------------

    print("Finished moving to task start configuration")
    # initialise communication socket to talk to policy:
    ctx = zmq.Context()
    socket = bind_zmq_socket(ctx, msg_address) 

    # action time is heuristic value mostly dependent on real world hardware
    # we found 0.4s was a good value too long and we get drift, too short and behaviour is jerky
    max_action_time = 0.4

    # check socket for initialising state request / reset request, initialise policy runner:
    initialise_insert_states(bolt_robot, socket)

    # command distributor waits for query/action input from policy runner, enacts action, responds with new state

    try:
        # I don't like the action output here
        pyaraas.run(pyaraas.Task(insertion_command_distributor, bolt_robot, bolt_f_sensor, max_action_time, socket))
    except Exception:
        traceback.print_exc()

    # when we have returned: terminate socket and context
    time.sleep(1.0)
    socket.close()
    ctx.term()

    # Do grasp release and cleanup
    #pyaraas.run(pyaraas.Task(strut_gripper.open, 50, 1.0))
    #strut_robot_cleanup_pose = Transform(buffer_pose)
    #strut_robot_cleanup_pose.position.y -= 150
    #strut_robot_cleanup_pose.position.x += 200
    #pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, buffer_pose, 1.0))
    #pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, strut_robot_cleanup_pose, 1.0))

    #pyaraas.run(pyaraas.Task(elbow_gripper.open, 50, 1.0))
    #elbow_robot_cleanup_pose = Transform(R1_ELBOW_PREGRASP)
    #elbow_robot_cleanup_pose.position.z += 120 # get well clear of the strut
    #pyaraas.run(pyaraas.Task(araas_small_move, elbow_robot, elbow_robot_cleanup_pose, 1.0))
    
    #pyaraas.run(pyaraas.Task(araas_small_move, elbow_robot, R1_ELBOW_PLANNED_POSE, 1.0))
