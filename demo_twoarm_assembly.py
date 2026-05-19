# test_rl_assembly
# wrapper and calling function for RL-based assembly tasks

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

config_name = "./bolt_gear_insertion_config.yaml" # Configuration file for the runner network.

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

GEAR_TOOL_OFFSET = RQ85_TCP_OFFSET 
BOLT_TOOL_OFFSET = RQ85_TCP_OFFSET # using same gripper on both robots

BOLT_GRASP_OFFSET = Transform(0.000000,0.000000,-5.000000,0.000000,0.000000,0.000000,1.000000) # offset between bolt origin and grasp - get this from CAD + chosen grasp position
BOLT_PICK_POSE = Transform(-5.000000,183.000000,41.200001, 1.000000,0.000000,0.000000,-0.000000).multiply(BOLT_GRASP_OFFSET) # pick pose is defined as the robot flange position in world coordinates when grasping - get this from digi twin or perception layer

# gear origin is at grasp origin, give or take - update this if we change models or grasp points
GEAR_GRASP_OFFSET = Transform(0, 0, 0,0, 0, 0)
GEAR_PICK_POSE = Transform(474.18,336.95,32.86, 0.000000,1.000000,-0.000000,0.000000) #.multiply(GEAR_GRASP_OFFSET) # gear-in-world pose

# goal poses before starting trained insertion policy

# define where we want the TCP to be in world coordinates:
GEAR_PRE_INSERT = Transform(50, 420, 421.7, 0.000000,-0.707107,0.000000,0.707107)
BOLT_PRE_INSERT = Transform(-52, 420, 435.7, 0.000000,0.707107,0.000000,0.707107)

# convert to robot flange transform:
# these are flipped somehow, ugh
BOLT_START_INSERT = BOLT_PRE_INSERT.multiply(BOLT_GRASP_OFFSET.multiply(BOLT_TOOL_OFFSET.invert())) #wait, where does grasp come in?
print(BOLT_GRASP_OFFSET.multiply(BOLT_TOOL_OFFSET.invert()).get_values())
print(BOLT_START_INSERT.get_values())
GEAR_START_INSERT = GEAR_PRE_INSERT.multiply(GEAR_TOOL_OFFSET.invert())


# GENERAL CONTROL CONSTRAINTS
# ------------------------------------

# heuristic thresholds, timesteps, gains, etc (these are usually applicable across multiple contexts)
translation_limit = 0.05 # (cap translation actions at 50mm)
rotation_limit = 1 / 180 *np.pi # in rad (only used if policy has rotational commands)

unit_scale = 1000 # controller gains are tuned for metres!! can either change gains, or just convert before sending
# (velocity command comes back in mm, for legacy reasons)

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

    # zero-ing and smoothing function might be messed up?

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

        # estimated forces still seem quite high :/


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

                except Exception:
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
    
    # need to scale from m to mm (could also just switch translation_limit to 50?)
    # p_delta is in metres, so convert to mm before calculating the goal differential

    p_delta = 1000*translation_limit*np.asarray(action) 
    r_delta = Quaternion(0,0,0,1) # null rotation if not controlling rotation

    # ARRGH! SIGN ERROR! this is a delta, not an error!

    p_start = robot.get_flange_transform().position
    p_goal = Vector(p_start.x + p_delta[0], p_start.y + p_delta[1], p_start.z+p_delta[2])
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
    rot_error = robot.get_flange_transform().rotation.multiply(goal_pose.rotation.invert()).get_rx_ry_rz() 

    pose_error = np.array([pos_error.x, pos_error.y, pos_error.z, rot_error.x, rot_error.y, rot_error.z])

    vel_est = pose_buffer.differential(robot.get_flange_transform()) # when we add velocity sampling, can get this directly
    # desired velocity is always zero:
    vel_error = vel_est - np.zeros(6)

    # Velocity error can get large but I think that's just registering actual velocity error
    
    ft_wrapper.update_force() # can we assume force has recently been initialised? if so, can probably take this out

    # check limits and clip the input sample if necessary:
    world_wrench = np.array(ft_wrapper.world_wrench)
    world_wrench_clip = controller.clip_input_wrench(world_wrench)

    # calculate a velocity command according to admittance control policy
    # I think we need these to be in m
    scaled_pose_error = 0.001*pose_error 
    scaled_vel_error = 0.001*vel_error
    
    linear_vel_cmd, rot_vel_cmd = controller.get_vel_cmd(world_wrench_clip, scaled_pose_error, scaled_vel_error, dt)

    # convert to robot flange pose and return:    
    robot_linear_vel = (pose_buffer.current_reading.rotation).invert().multiply(linear_vel_cmd)
    robot_rot_vel = ((pose_buffer.current_reading.rotation).invert().multiply(rot_vel_cmd))

    return robot_linear_vel, robot_rot_vel


def initialise_policy_states(robot0, robot1,  socket):
    action_msg = socket.recv_json()

    if 'update' in action_msg:
        logging.info("received update state request")

        r0_ee_pose = robot0.get_flange_transform()
        r1_ee_pose = robot1.get_flange_transform()

        r0_trans = np.array([r0_ee_pose.position.x, r0_ee_pose.position.y, r0_ee_pose.position.z])
        r0_rot = np.array(r0_ee_pose.extract_row_major()).reshape((4,4))

        r1_trans = np.array([r1_ee_pose.position.x, r1_ee_pose.position.y, r1_ee_pose.position.z])
        r1_rot = np.asarray(r1_ee_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r0_state_return = {}
        r1_state_return = {}

        r0_state_return["trans"] = (r0_trans).tolist()
        r0_state_return["rot_c1"] = (np.array(r0_rot[:3, 0])).tolist()
        r0_state_return["rot_c2"] = (np.array(r0_rot[:3, 1])).tolist()
        
        state_message["r0_pose"] = r0_state_return

        r1_state_return["trans"] = (r1_trans).tolist()
        r1_state_return["rot_c1"] = (np.array(r1_rot[:3, 0])).tolist()
        r1_state_return["rot_c2"] = (np.array(r1_rot[:3, 1])).tolist()

        state_message["r1_pose"] = r1_state_return
        socket.send_json(state_message) 


    else:
        print("Unexpected message recieved from socket")
        print(action_msg)
        logging.info("Initialization error")


async def insertion_command_distributor(robot0, robot1, ft0, ft1, action_time, socket):

    logging.info("launching robot control updater")
    # set up buffer structures for forces, velocities, etc (velocity currently unused)

    # initialise buffers:
    r0_flange_velocity = Transform()
    r1_flange_velocity = Transform()
    r0_pose_init = robot0.get_flange_transform()
    r1_pose_init = robot0.get_flange_transform()

    r0_pose_buffer = SampleBuffer(r0_pose_init, dt)
    r1_pose_buffer = SampleBuffer(r1_pose_init, dt)

    # we actually don't use these
    r0_vel_buffer = SampleBuffer(r0_flange_velocity, dt)
    r1_vel_buffer = SampleBuffer(r1_flange_velocity, dt)

    # initialise force sensor data wrappers
    history_size = 8 # we use a history buffer as a moving window to smooth force sensor readings - this could also be built into araas
    f0_data = ForceSensorWrapper(robot0, ft0, history_size)
    f1_data = ForceSensorWrapper(robot1, ft1, history_size)

    # when using OnRobot sensors, as long is the load is light we can sample load-compensated readings and eliminate the bias zeroing
    # (this also allows better control of rotational dimensions)

    f0_data.calculate_sensor_bias()
    f1_data.calculate_sensor_bias()

    # make sure we can read from sensor:
    f0_data.update_force()
    f1_data.update_force()

    # Initialise controllers 
    mass_matrix = np.array([M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_LIN, M_DEFAULT_ANG, M_DEFAULT_ANG, M_DEFAULT_ANG])
    kp_matrix = np.array([KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT, KP_DEFAULT])
    kd_matrix = np.array([KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_LIN, KD_DEFAULT_ANG, KD_DEFAULT_ANG, KD_DEFAULT_ANG])

    r0_admit_control = AdmittanceController(mass_matrix, kp_matrix, kd_matrix, force_limit, torque_limit, 1000, t_vel_limit, r_vel_limit)
    r1_admit_control = AdmittanceController(mass_matrix, kp_matrix, kd_matrix, force_limit, torque_limit, 1000, t_vel_limit, r_vel_limit)

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
        robot_0_action = action_msg["r0_action"]
        robot_1_action = action_msg["r1_action"]

        robot_0_goal = extract_action(robot0, robot_0_action)
        robot_1_goal = extract_action(robot1, robot_1_action)
        print("Incoming command:")
        print(robot_0_goal.get_values())
        print(robot_1_goal.get_values())

        init_time = time.time()
    
        while((time.time() - init_time) < action_time): 
            #if we go too fast, we don't have time to converge on the goal speed
            r0_lin_cmd, r0_rot_cmd = get_control_update(robot0, f0_data,r0_admit_control, robot_0_goal,r0_pose_buffer)
            r1_lin_cmd, r1_rot_cmd = get_control_update(robot1, f1_data, r1_admit_control, robot_1_goal, r1_pose_buffer)

            # directions seem off, hmm
            print("Checking dimensional split of controls")
            print(r0_lin_cmd)
            print(r1_lin_cmd)

            # send commands to hardware
            _, r0_pose_update = robot0.set_cartesian_velocity(r0_lin_cmd, r0_rot_cmd, dt, force_limit, mass=1)
            _, r1_pose_update = robot1.set_cartesian_velocity(r1_lin_cmd, r1_rot_cmd, dt, force_limit, mass=1)

            # not entirely sure why control delay/state update is structured like this, could be cleaner
            # are we just too fast, maybe?
            time.sleep(dt/2) 

            print("Checking time elapsed")
            print(time.time()-init_time)

            r0_pose_buffer.update(r0_pose_update)
            r1_pose_buffer.update(r1_pose_update)

            # update velocity, though it's not really used just now (vel is approximated from pose delta when we need it)
            new_vel_r0 = r0_pose_buffer.differential(r0_pose_update)
            new_vel_r1 = r1_pose_buffer.differential(r1_pose_update)

            r0_vel_buffer.update(Transform(new_vel_r0))
            r1_vel_buffer.update(Transform(new_vel_r1))

            f0_data.update_force()
            f1_data.update_force()


        # construct observation message:
        logging.info("Finished control action")

        # Observation (in this training regime/checkpoint) is just the flange pose. Other networks may 
        # have more information attached to obs.

        r0_ee_pose = robot0.get_flange_transform()
        r1_ee_pose = robot1.get_flange_transform()

        r0_trans = np.array([r0_ee_pose.position.x, r0_ee_pose.position.y, r0_ee_pose.position.z])
        r0_rot = np.array(r0_ee_pose.extract_row_major()).reshape((4,4))

        r1_trans = np.array([r1_ee_pose.position.x, r1_ee_pose.position.y, r1_ee_pose.position.z])
        r1_rot = np.asarray(r1_ee_pose.extract_row_major()).reshape((4,4))

        # extract first two columns of rotation matrix to represent pose
        state_message = {}
        r0_state_return = {}
        r1_state_return = {}

        r0_state_return["trans"] = (r0_trans).tolist()
        r0_state_return["rot_c1"] = (np.array(r0_rot[:3, 0])).tolist()
        r0_state_return["rot_c2"] = (np.array(r0_rot[:3, 1])).tolist()
        
        state_message["r0_pose"] = r0_state_return

        r1_state_return["trans"] = (r1_trans).tolist()
        r1_state_return["rot_c1"] = (np.array(r1_rot[:3, 0])).tolist()
        r1_state_return["rot_c2"] = (np.array(r1_rot[:3, 1])).tolist()

        state_message["r1_pose"] = r0_state_return
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

    peg_robot = workcell.get_robot("UR10e-1") #check these 
    hole_robot = workcell.get_robot("UR10e-0")
    peg_gripper = workcell.get_gripper("Rq85-1")
    hole_gripper = workcell.get_gripper("Rq85-0")

    # get force sensors
    peg_f_sensor = workcell.get_force_sensor('UR10e-1')
    hole_f_sensor = workcell.get_force_sensor('UR10e-0')

    # go to workcell default start poses
    print("Joint driving to safe poses to eliminate windup") # this plans joint-to-joint motion, while checking for collisions

    try:
        if not (peg_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)) and not (hole_robot.get_flange_transform().is_equal(0.1, 0.01, WORLD_T_R0_HOME)):
            pyaraas.run([pyaraas.Task(joint_path_execute, workcell, peg_robot, SAFE_JOINT_1, 5.0), pyaraas.Task(joint_path_execute, workcell, hole_robot, SAFE_JOINT_0, 5.0)])

        elif not (peg_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R1_HOME)):
            pyaraas.run(pyaraas.Task(joint_path_execute, workcell, peg_robot, SAFE_JOINT_1, 5.0))

        elif not (hole_robot.get_flange_transform().is_equal(5, 0.03, WORLD_T_R0_HOME)):
                pyaraas.run(pyaraas.Task(joint_path_execute, workcell, hole_robot, SAFE_JOINT_0, 5.0))

    except Exception:
        traceback.print_exc()
        import sys
        sys.exit()
        
    # do grasping and pickup
    # create an offset transform for picking items off the kit:
    approach = Transform([0, 0, 100, 0, 0, 0])

    # (should not need to actively set collision models right after initialisation, but can do so just as a failsafe)
    peg_robot.set_collision_model(True, True)
    peg_gripper.set_collision_model(True, True)
    hole_robot.set_collision_model(True, True)
    hole_gripper.set_collision_model(True, True)

    # pick up bolt, pick up gear
    gear_grasp_pose = GEAR_PICK_POSE.multiply(GEAR_TOOL_OFFSET.invert())
    gear_pregrasp_pose = approach.multiply(gear_grasp_pose)

    bolt_grasp_pose = BOLT_PICK_POSE.multiply(BOLT_TOOL_OFFSET.invert())
    bolt_pregrasp_pose = approach.multiply(bolt_grasp_pose)

    try:
        pyaraas.run([pyaraas.Task(araas_path_planner, workcell, hole_robot, gear_pregrasp_pose, 3.0),
                     pyaraas.Task(araas_path_planner, workcell, peg_robot, bolt_pregrasp_pose, 3.0)])
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit()

    print("Picking up")

    # Collision disabling code block, to avoid spurious kit/table/taskboard collisions
    # ----------------------------
    hole_robot.set_collision_model(False, False)
    hole_gripper.set_collision_model(False, False)
    peg_robot.set_collision_model(False, False)
    peg_gripper.set_collision_model(False, False)
    # ----------------------------

    # move down to grasp point(s), perform grasp(s), move back up to pre-grasp pose
    # 'small move' performs a linear cartesian motion in robot flange space

    pyaraas.run([pyaraas.Task(araas_small_move, hole_robot, gear_grasp_pose, 1.0), pyaraas.Task(araas_small_move, peg_robot, bolt_grasp_pose, 1.0)])
    pyaraas.run([pyaraas.Task(hole_gripper.open, 0, 1), pyaraas.Task(peg_gripper.open, 0, 1)])

    # because the bolt is quite long, add an additional buffer when pulling back to ensure we're well clear of the kit
    pregrasp_adjust = Transform(bolt_pregrasp_pose)
    pregrasp_adjust.position.x -= 300 
    pyaraas.run([pyaraas.Task(araas_small_move, hole_robot, gear_pregrasp_pose, 1.0), pyaraas.Task(araas_small_move, peg_robot, bolt_pregrasp_pose, 1.0)])
    pyaraas.run(pyaraas.Task(araas_small_move, peg_robot, pregrasp_adjust, 1.5))

    print("Completed bolt and gear pickup")

    print("Moving to pre-insertion positions")

    # enable collisions for large motions
    # -----------------------------
    hole_robot.set_collision_model(True, True)
    hole_gripper.set_collision_model(True, True)
    peg_robot.set_collision_model(True, True)
    peg_gripper.set_collision_model(True, True)
    # -----------------------------

    # performing pre-insertion motions one at a time to avoid collision mid-motion (as planner has static workcell state)
    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, hole_robot, GEAR_START_INSERT, 3.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 

    # HEURISTIC ADJUSTMENT: start points are close together, so plan bolt motion to an approach offset, not directly to target
    pre_insertion = Transform(BOLT_START_INSERT)
    pre_insertion.position.x -= 100

    try:
        pyaraas.run(pyaraas.Task(araas_path_planner, workcell, peg_robot, pre_insertion, 3.0))
        pyaraas.run(pyaraas.Task(araas_small_move, peg_robot, BOLT_START_INSERT, 1.0))
    except Exception:
        print(traceback.format_exc())
        import sys
        sys.exit() 

    # Collision disabling code block, necessary before running an insertion policy
    # ----------------------------
    hole_robot.set_collision_model(False, False)
    hole_gripper.set_collision_model(False, False)
    peg_robot.set_collision_model(False, False)
    peg_gripper.set_collision_model(False, False)
    # ----------------------------

    print("Finished moving to task start configuration")
    # initialise communication socket to talk to policy:
    ctx = zmq.Context()
    socket = bind_zmq_socket(ctx, msg_address) 

    # action time is heuristic value mostly dependent on real world hardware
    # we found 0.4s was a good value too long and we get drift, too short and behaviour is jerky
    max_action_time = 0.4

    # check socket for initialising state request / reset request, initialise policy runner:
    initialise_policy_states(hole_robot, peg_robot, socket)

    # command distributor waits for query/action input from policy runner, enacts action, responds with new state
    try:
        pyaraas.run(pyaraas.Task(insertion_command_distributor, hole_robot, peg_robot, hole_f_sensor, peg_f_sensor, max_action_time, socket))
    except Exception:
        traceback.print_exc()
        import sys
        sys.exit()

    # when we have returned: terminate socket and context
    socket.close()
    ctx.term()

    # Do grasp release and cleanup (TODO add this)



