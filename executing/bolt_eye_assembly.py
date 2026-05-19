# Variables: enable hardware can be on or off
# Starting point: very bare bones, use two-handed insertion as a test environment, no sequencing or looping

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

import json

import traceback
import logging 

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from rl_games.common import env_configurations # check rl games env requirements
    from rl_games.torch_runner import Runner

from rl_games.algos_torch import model_builder

config_name = "./bolt_eye_insertion_config.yaml" # Configuration file for the runner network.

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

RQ85_TCP_OFFSET = Transform(0,0,300,0,0,0,1) # distance between robot wrist and gripper TCP point. 
# We can also pull this from araas - see define_offset_transform function. Hardcoded for transparency for now.

# 1) Which robot is holding strut and which is holding bolt? 
# 2) Should this script also do strut-elbow first? Preferably not - makes it too annoying to debug. So, how to do set up?
# - check which robot will be 'holding' the bolt and which the strut
# - for debugging, have this in mid-air - basically a weirder version of rod/gear. Can we tighten up the gain/lose some spring on the strut robot?
# - when to bring it to the platform? Depends on success - if we can't get it working in mid-air, don't waste time, just pivot to a retrain
# - retrain effort should be a SINGLE insertion, but with a (rotational) admittance control on the strut robot which tries to stay in the same position
# as start, with some initial springiness.


STRUT_TOOL_OFFSET = RQ85_TCP_OFFSET 
BOLT_TOOL_OFFSET = RQ85_TCP_OFFSET # using same gripper on both robots

BOLT_GRASP_OFFSET = Transform(0.000000,0.000000,16.000000,0.000000,0.000000,0.000000,1.000000) # offset between bolt origin and grasp - get this from CAD + chosen grasp position
BOLT_PICK_POSE = Transform(-5.000000,183.000000,41.200001, 1.000000,0.000000,0.000000,-0.000000).multiply(BOLT_GRASP_OFFSET) # pick pose is defined as the robot flange position in world coordinates when grasping - get this from digi twin or perception layer

# offset between strut origin and chosen grasp:
STRUT_GRASP_OFFSET = Transform(22.0, -5.0, 26.0, 1.5707963705062866, 1.5707963705062866, 0.0)
STRUT_PICK_POSE = Transform(40.000, 401.1999, 24.0, 0.0, 1.570, 0.0) #strut-in-world pose

# goal poses before starting trained insertion policy

# define where we want the TCP to be in world coordinates:

STRUT_PRE_INSERT = Transform(50, 420, 320, 1.57079, -1.57059, 0) #Transform(76.00, 120.0, 299.700, -3.141592, 0.0, -1.57079)

# might need to adjust this (esp. height) to accomodate flange-to-eye offset in other robot
BOLT_PRE_INSERT = Transform(-9, 420, 435.7, 0.000000,0.707107,0.000000,0.707107)

# convert to robot flange transform:

BOLT_START_INSERT = BOLT_PRE_INSERT.multiply(BOLT_TOOL_OFFSET.invert())
STRUT_START_INSERT = STRUT_PRE_INSERT.multiply(STRUT_TOOL_OFFSET.invert()) #this looks like it's already hard coded??


# GENERAL CONTROL CONSTRAINTS
# ------------------------------------

# heuristic thresholds, timesteps, gains, etc (these are usually applicable across multiple contexts)
translation_limit = 0.05 # (cap translation actions at 50mm)
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
                self.torque = self.sample["torque"]
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

        new_world_force = (self.robot.get_flange_transform().multiply(self.force)).subtract(force_bias)
        new_world_torque = (self.robot.get_flange_transform().multiply(self.torque)).subtract(torque_bias)

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
                    self.torque = self.sample["torque"]
                except:
                    print("Reading from sensor failed")
                    self.force = Vector(0,0,0)
                    self.torque = Vector(0,0,0)

            else:
                # dummy read
                self.force = Vector(0,0,0)
                self.torque = Vector(0,0,0)

            new_world_force = self.robot.get_flange_transform().multiply(self.force)
            new_world_torque = self.robot.get_flange_transform().multiply(self.torque)

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
        return (1/self.dt)*delta


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
    p_delta = translation_limit*np.asarray(action)
    r_delta = Quaternion(0,0,0,1) # null rotation if not controlling rotation

    p_start = robot.get_flange_transform().position
    p_goal = Vector(p_start.x - p_delta[0], p_start.y - p_delta[1], p_start.z-p_delta[2])
    r_goal = robot.get_flange_transform().rotation.multiply(r_delta)

    goal_pose = Transform(p_goal.x, p_goal.y, p_goal.z,r_goal.x, r_goal.y, r_goal.z, r_goal.w)
    logging.info("Goal pose calculated")

    return goal_pose


def get_control_update(robot, 
        ft_wrapper, 
        controller,
        goal_pose, 
        pose_buffer):

    # TO DO: add velocity reading from driver (or velocity calculation at wrapper level instead of python level!)
    pos_error = robot.get_flange_transform().position.subtract(goal_pose.position)
    rot_error = (robot.get_flange_transform().rotation.multiply(goal_pose.rotation)).get_rx_ry_rz()

    pose_error = np.array([pos_error.x, pos_error.y, pos_error.z, rot_error.x, rot_error.y, rot_error.z])

    vel_est = pose_buffer.differential(robot.get_flange_transform()) # when we add velocity sampling, can get this directly

    # desired velocity is always zero:
    vel_error = vel_est - np.zeros(6)

    ft_wrapper.update_force() # can we assume force has recently been initialised? if so, can probably take this out

    # check limits and clip the input sample if necessary:
    world_wrench = np.array(ft_wrapper.world_wrench)
    world_wrench_clip = controller.clip_input_wrench(world_wrench)

    # calculate a velocity command according to admittance control policy
    linear_vel_cmd, rot_vel_cmd = controller.get_vel_cmd(world_wrench_clip, pose_error, vel_error, dt)

    # convert to robot flange pose and return:    
    robot_linear_vel = (pose_buffer.current_reading.rotation).invert().multiply(linear_vel_cmd)
    robot_rot_vel = ((pose_buffer.current_reading.rotation).invert().multiply(rot_vel_cmd))

    return robot_linear_vel, robot_rot_vel

def initialise_policy_states(robot0, socket):
    action_msg = socket.recv_json()

    if 'update' in action_msg:
        logging.info("received update state request")

        r0_ee_pose = robot0.get_flange_transform()

        r0_trans = np.array([r0_ee_pose.position.x, r0_ee_pose.position.y, r0_ee_pose.position.z])
        r0_rot = np.array(r0_ee_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r0_state_return = {}

        r0_state_return["trans"] = (r0_trans).tolist()
        r0_state_return["rot_c1"] = (np.array(r0_rot[:3, 0])).tolist()
        r0_state_return["rot_c2"] = (np.array(r0_rot[:3, 1])).tolist()
        
        state_message["robot_pose"] = r0_state_return

        socket.send_json(state_message) 


    else:
        print("Unexpected message recieved from socket")
        print(action_msg)
        logging.info("Initialization error")


async def insertion_command_distributor(robot, ft, action_time, socket):

    logging.info("launching robot control updater")
    # set up buffer structures for forces, velocities, etc (velocity currently unused)

    # initialise buffers:
    r_flange_velocity = Transform()
    r_pose_init = robot.get_flange_transform()

    r_pose_buffer = SampleBuffer(r_pose_init, dt)

    # we actually don't use these
    r_vel_buffer = SampleBuffer(r_flange_velocity, dt)

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

        robot_goal = extract_action(robot, robot_action)

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

            r_vel_buffer.update(Transform(new_vel))
            f_data.update_force()


        # construct observation message:
        logging.info("Finished control action")

        # Observation (in this training regime/checkpoint) is just the flange pose. Other networks may 
        # have more information attached to obs.

        r_ee_pose = robot.get_flange_transform()

        r_trans = np.array([r_ee_pose.position.x, r_ee_pose.position.y, r_ee_pose.position.z])
        r_rot = np.array(r_ee_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r_state_return = {}
        
        r_state_return["trans"] = (r_trans).tolist()
        r_state_return["rot_c1"] = (np.array(r_rot[:3, 0])).tolist()
        r_state_return["rot_c2"] = (np.array(r_rot[:3, 1])).tolist()
        
        state_message["robot_pose"] = r_state_return

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
    # Peg should be === bolt, hole should be === gear. Gear is r0, so ...

    bolt_robot = workcell.get_robot("UR10e-1") #check these 
    strut_robot = workcell.get_robot("UR10e-0")
    bolt_gripper = workcell.get_gripper("Rq85-1")
    strut_gripper = workcell.get_gripper("Rq85-0")

    # get force sensors
    bolt_f_sensor = workcell.get_force_sensor('UR10e-1')
    strut_f_sensor = workcell.get_force_sensor('UR10e-0')

    # go to workcell default start poses
    print("Joint driving to safe poses to eliminate windup") # this plans joint-to-joint motion, while checking for collisions

    try:
        if not (bolt_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)) and not (strut_robot.get_flange_transform().is_equal(0.1, 0.01, WORLD_T_R0_HOME)):
            pyaraas.run([pyaraas.Task(joint_path_execute, workcell, bolt_robot, SAFE_JOINT_1, 5.0), pyaraas.Task(joint_path_execute, workcell, strut_robot, SAFE_JOINT_0, 5.0)])

        elif not (bolt_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)):
            pyaraas.run(pyaraas.Task(joint_path_execute, workcell, bolt_robot, SAFE_JOINT_1, 5.0))

        elif not (strut_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R0_HOME)):
                pyaraas.run(pyaraas.Task(joint_path_execute, workcell, strut_robot, SAFE_JOINT_0, 5.0))

    except Exception:
        traceback.print_exc()
        import sys
        sys.exit()
        
    # do grasping and pickup
    # create an offset transform for picking items off the kit:
    approach = Transform([0, 0, 100, 0, 0, 0])

    # (should not need to actively set collision models right after initialisation, but can do so just as a failsafe)
    bolt_robot.set_collision_model(True, True)
    bolt_gripper.set_collision_model(True, True)
    strut_robot.set_collision_model(True, True)
    strut_gripper.set_collision_model(True, True)

    # pick up bolt, pick up gear    
    strut_robot_pick_pose = STRUT_PICK_POSE.multiply(STRUT_GRASP_OFFSET) #gear offset was null so this got skipped before
    strut_grasp_pose = strut_robot_pick_pose.multiply(STRUT_TOOL_OFFSET.invert())
    strut_pregrasp_pose = approach.multiply(strut_grasp_pose)

    print("check pregrasp of strut ")
    print(strut_pregrasp_pose.get_values())

    bolt_grasp_pose = BOLT_PICK_POSE.multiply(BOLT_TOOL_OFFSET.invert())
    bolt_pregrasp_pose = approach.multiply(bolt_grasp_pose)

    # split up pick up actions, because they're quite close together

    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, strut_robot, strut_pregrasp_pose, 3.0)),
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit()

    print("Picking up")

    # Collision disabling code block, to avoid spurious kit/table/taskboard collisions
    # ----------------------------
    strut_robot.set_collision_model(False, False)
    strut_gripper.set_collision_model(False, False)
    # ----------------------------

    # move down to grasp point(s), perform grasp(s), move back up to pre-grasp pose
    # 'small move' performs a linear cartesian motion in robot flange space

    pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, strut_grasp_pose, 1.0))
    pyaraas.run(pyaraas.Task(strut_gripper.open, 0, 1))

    # will need to pivot - move to a clear point first:
    strut_reorient_pose = Transform(strut_pregrasp_pose)
    strut_reorient_pose.position.x = 310
    strut_reorient_pose.position.z = 450

    pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, strut_pregrasp_pose, 1.0)) 
    pyaraas.run(pyaraas.Task(araas_small_move, strut_robot, strut_reorient_pose, 1.0)) 

    # Now pick up bolt:
    bolt_robot.set_collision_model(False, False)
    bolt_gripper.set_collision_model(False, False)
    
    pyaraas.run(pyaraas.Task(araas_path_planner, workcell, bolt_robot, bolt_pregrasp_pose, 3.0))

    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, bolt_grasp_pose, 1.0))
    pyaraas.run(pyaraas.Task(bolt_gripper.open, 0, 1))

    # because the bolt is quite long, add an additional buffer when pulling back to ensure we're well clear of the kit
    pregrasp_adjust = Transform(bolt_pregrasp_pose)
    pregrasp_adjust.position.x -= 300 
    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, bolt_pregrasp_pose, 1.0))
    pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, pregrasp_adjust, 1.5))

    print("Completed bolt and gear pickup")

    print("Moving to pre-insertion positions")

    # enable collisions for large motions
    # -----------------------------
    strut_robot.set_collision_model(True, True)
    strut_gripper.set_collision_model(True, True)
    bolt_robot.set_collision_model(True, True)
    bolt_gripper.set_collision_model(True, True)
    # -----------------------------

    # performing pre-insertion motions one at a time to avoid collision mid-motion (as planner has static workcell state)

    print("moving to strut alignment pose")
    print(STRUT_START_INSERT.get_values())


    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, strut_robot, STRUT_START_INSERT, 3.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 

    # HEURISTIC ADJUSTMENT: start points are close together, so plan bolt motion to an approach offset, not directly to target
    pre_insertion = Transform(BOLT_START_INSERT)
    pre_insertion.position.x -= 100

    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, bolt_robot, pre_insertion, 3.0))
        pyaraas.run(pyaraas.Task(araas_small_move, bolt_robot, BOLT_START_INSERT, 1.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 

    # Collision disabling code block, necessary before running an insertion policy
    # ----------------------------
    strut_robot.set_collision_model(False, False)
    strut_gripper.set_collision_model(False, False)
    bolt_robot.set_collision_model(False, False)
    bolt_gripper.set_collision_model(False, False)
    # ----------------------------

    print("Finished moving to task start configuration")
    # initialise communication socket to talk to policy:
    ctx = zmq.Context()
    socket = bind_zmq_socket(ctx, msg_address) 

    # action time is heuristic value mostly dependent on real world hardware
    # we found 0.4s was a good value too long and we get drift, too short and behaviour is jerky
    max_action_time = 0.4

    # check socket for initialising state request / reset request, initialise policy runner:
    initialise_policy_states(bolt_robot, socket)

    # command distributor waits for query/action input from policy runner, enacts action, responds with new state
    try:
        pyaraas.run(pyaraas.Task(insertion_command_distributor,  bolt_robot, bolt_f_sensor, max_action_time, socket))
    except Exception:
        traceback.print_exc()
        import sys
        sys.exit()

    # when we have returned: terminate socket and context
    socket.close()
    ctx.term()

    # Do grasp release and cleanup (TODO add this)



