import pyatk
import pyaraas
from pyatk import Transform, Vector, Quaternion, Part, Actor, Gripper, Articulation, PlanningError
from pyaraas.tools import PathPlanner 

from admittance_controller import AdmittanceController

import time
import numpy as np
import os, yaml,argparse

import math
import zmq
import copy

import json


# some useful transform constants, probably not needed

RQ85_TCP_OFFSET = Transform(0,0,293,0,0,0,1) 
STRUT_TOOL_OFFSET = RQ85_TCP_OFFSET 

# offset between strut origin and chosen grasp:
STRUT_GRASP_OFFSET = Transform(22.0, -5.0, 26.0, 1.5707963705062866, 1.5707963705062866, 0.0)
STRUT_PICK_POSE = Transform(40.000, 401.1999, 24.0, 0.0, 1.570, 0.0) #strut-in-world pose

STRUT_PRE_INSERT = Transform(50, 420, 320, 1.57079, -1.57059, 0) #Transform(76.00, 120.0, 299.700, -3.141592, 0.0, -1.57079)
STRUT_START_INSERT = STRUT_PRE_INSERT.multiply(STRUT_TOOL_OFFSET.invert())

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


    # pick up bolt, pick up gear    
    strut_robot_pick_pose = STRUT_PICK_POSE.multiply(STRUT_GRASP_OFFSET) #gear offset was null so this got skipped before
    strut_grasp_pose = strut_robot_pick_pose.multiply(STRUT_TOOL_OFFSET.invert())
    strut_pregrasp_pose = approach.multiply(strut_grasp_pose)
