
import json
import logging
from typing import List, Tuple, Dict
import os
import time
import pathlib
import numpy as np
from dataclasses import dataclass, field
import random
import pybullet as p
import numpy as np 
import math
from pathlib import Path
import numpy as np
import torch
import pb_utils as pbu
import time
import copy

BASE_LINK = -1
ROOT_DIR = Path(__file__).parent.parent

# Process related functions (eg logs, etc) are left in utils 
# ISAACGYM and ARAAS pull from the same core task config class, so it's fine to have them together, I think. 
# we just need to ensure that there isn't any specific library cross over (eg pathlib/Path - what is this?)

# we just need to 1) clearly organise constants and functions, and 2) leave anything that doesn't touch either environment
# in the utils file

# GRASPS are defined either RELATIVE TO ORIGIN or RELATIVE TO INSERTION POINT - bring these into alignment
# and make any necessary corrections to perception pipeline data. 
# (often origin / insertion points are co-located, but they do not have to be!!! Has this been accounted for on the backend?)

# Part relative transforms SHOULD be defined relative to part origin, but see above. 

# PEG is the object with the relevant assembly extrusion, and sub transforms for individual pegs
# are defined relative to this object



# ---------------- ENVIRONMENT OVERVIEW ------------------#
# Environment assumptions: workcell has two robots, task is a generalized "peg in hole" function, 
# either a single robot can be used to insert a peg into a fixed hole, 
# or one robot can grasp the 'peg' and one can grasp the 'hole'.

# ----------------- SHARED CONSTANTS ---------------------#

# Many of these are environment-specific, and named as such - see the TaskConfig generation functions for
# how to convert specific offsets, etc, into generalized target transforms.


# ---------------- Fixed transforms  ----------------#
# Transform between ee base frame and TCP point: note that if the robots are not using matched grippers,
# this needs to be accounted for here and in the running scripts

TOOL_T_TIP = ([0, 0, 0.30], [0, 0, 0, 1]) # offset for R85 (I think?) We should be able to pull from araas, maybe the info was wrong?
# or possibly just used for training environment

# --------------- NIST data ---------------#
# For speedy adjustment of NIST-board tasks, we use some helper transforms so that we can localise
# parts within the taskboar, rather than absolutely against the workcell origin. These aren't relevant for
# the general case.

TASKBOARD_POSE = ([0.150, 0.315, -0.034], [0.7071068, 0, 0, 0.7071068]) # NIST taskboard location

# Part poses relative to NIST taskboard origin
PLATFORM_T_MEDGEAR = ([0.2172, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17]) #([0.2172, 0.0617, 0.2647]
PLATFORM_T_SMALLGEAR =  ([0.1872, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
PLATFORM_T_LARGEGEAR = ([0.2672, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])

# ---------------- Two-robot bolt/gear insertion ---------------#
BOLT_PICK_POSE = ([-0.35973, 0.25, 0.290], [-1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.3267949e-06]) # pick up bolt
GEAR_PICK_POSE = ([0.640, 0.11735, 0.270], [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.3267949e-06]) # for bolt-in-gear tests


# ---------------- Stewart platform -------------#
# similar helper transforms specifically for stewart platform assembly. 
# These can (potentially) be discarded as we move to a perception-in-the-loop practice, although
# could be useful to initialise camera poses

PLATFORM_POSE = ([[0.005, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]]) # Stewart platform origin
WORLD_T_KIT = ([-0.005, 0.288, 0.011], [0, 0, -0.70711, 0.70711]) #Kit position in workcell
KIT_T_ELBOW_0 = ([0.09273, -0.07578, -0.01149], [0, -0.70711, 0.70711, 0]) # internal kitting transforms
# transform: rotate 90 degrees around ELBOW COMPONENT X axis and 180 degrees around (ELBOW COMPONENT?) Z axis - order of operations matters.

KIT_T_STRUT_5 = ([-0.1132, 0.045, 0.013], [-0.5, 0.5, 0.5, 0.5])
KIT_T_BOLT_0 = ([0.105, 0, 0.0302],[0.70711, 0.70711, 0, 0])

# Hard-coded grasp transforms - replace with generated grasp!!

# These are all defined relative to feature of interest (eg peg, hole, ...)
# so automatically give us the (tool tip)->(feature) transform
# strut grasp is 32mm above node? this doesn't give us much to play with

# moving the strut grasp doesn't really help with insertion collision, but also, it may not be a big deal in real
# THAT SAID, we have to enable this somehow.
# Regrasping the strut from a different angle may be the best choice.

STRUT_GRASP = ([0.022, -0.005, 0.026], [0.5, 0.5, 0.5, 0.5]) # bit loose, adjusting -5mm on the Y
BOLT_GRASP = ([0, 0, -0.005], [0, 0, 0, 1]) #[0, 0, 0.70711, 0.70711]) 
ELBOW_GRASP = ([-0.0195, 0.00769, -0.02], [0,0.70711, 0.70711, 0]) # [-0.8525245, 0, 0,0.5226872]

ELBOW_REGRASP = ([-0.135,0.553825317,0.216767059], [0.352823,0.610489,0.614080,0.354577])
 # hard coded regrasp to enable straight strut insertion! need to generate on the fly!

# transform: rotate -90 degrees around ELBOW COMPONENT X axis.

# Goal points for insertion, relative to stewart platform base:
# using insertion point close to (r1) causes configuraton issues for elbow insertion
# using insertion point close to (r0) causes collision issues if a wrist camera is mounted
# either could be fixed by a grasp realignment. The latter could possibly be fixed by an orientation flip?

# Trying to re-orient grasp / kit initialisation point is difficult. Cannot edit elbow-in-platform pose - big problems!

# OK NOW WE ARE GETTING SOMEWHERE - Bullet seems to be solving IKs on its own in the backend???
# and then we get weird collisions
# Why is bullet being engaged at this stage at all? Surely it would be cleaner to boot it up with (starting conditions)
# right before running the learned task policy, and shut it down afterwards?

# also, which poses can and cannot be edited without messing up the back end is unclear. 

PLATFORM_T_STRUT = ([[0.0002, 0.15019, 0.23461], [0, 0, -0.88691, 0.46195]])    # (0.0002, 0.15019, 0.23461)
PLATFORM_T_ELBOW = ([-0.01656, 0.16937, 0.20586], [0, 0, 0.70711, 0.70711])     # (-0.1656, 0.16937, 0.20586)
PLATFORM_T_BOLT = ([[-0.00792, 0.0856, 0.41553], [0.62253, 0.33534, -0.62253, 0.33534]]) #(-0.00792, 0.0856, 0.41553)
STRUT_T_BOLT = ([[0, 0, 0.20243], [0.5, 0.5, 0.5, 0.5]]) # try this [0, 0.70711, 0, 0.70711]

# ------------------ Screw and bearing  -------------#
# (I think? why is this 'wrench'? confusing)
BEARING_GRASP = ([0, -0.01658, 0], [0.70711, 0, 0, 0.70711])
WRENCH_GRASP = ([0, 0.013, -0.01768], [0, -1, 0, 0])
WRENCH_T_BEARING_GOAL = ([0, 0.08578, 0], [0, 0, 0, 1])
WRENCH_T_BEARING_START = ([0.04146, 0.10988, 0], [0,0, -0.70711, 0.70711])

# ---- possibly unused
FLANGE_T_TOOL = ([0,0,0], [0.5, 0.5, 0.5, 0.5]) # is this a 180 degree flip? Sort of - it's a 90/90 x/y flip, which might achieve the same purpose


# ---------------- TRAINING ENVIRONMENT ------------------#

# If using a non-araas environment (eg ISAACgym), there is additional setup info that comes for free in the araas workcell config
# eg. workcell base poses for the two robots, initial joint configurations, and collision tuples to ignore (internal collisions)

R0_POSE = ((0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, 0.7071068, 0.7071068))
R1_POSE = ((-0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, -0.7071068, 0.7071068))

DEFAULT_R0_CONF = [-0.5144105923545090, -1.643108558589230, -2.105897059942356, -0.9535990805992884, 1.561290095498969, -0.5144860744476318]
DEFAULT_R1_CONF = [-0.9920667074360754, -1.107338840559089, -2.473368842697666, -1.128834397685560, 1.563770470523882, 0.01950128190219402]
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

IGNORE_COLLISIONS = {(6, 9), (2, 10), (3, 10), (4, 10), (6, 10), (3, 9), (4, 9), (2, 11), (3, 11), (5, 11), (6, 11), (10, 11), (2, 12), (9, 10)}


def setup_environment(ee_collisions=True):
    if(ee_collisions):
        r0 = p.loadURDF("ur_description/urdf/ur10e.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
        r1 = p.loadURDF("ur_description/urdf/ur10e.urdf", R1_POSE[0], R1_POSE[1], useFixedBase=True)
    else:
        r0 = p.loadURDF("ur_description/urdf/ur10e_no_hand.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
        r1 = p.loadURDF("ur_description/urdf/ur10e_no_hand.urdf", R1_POSE[0], R1_POSE[1], useFixedBase=True)

    # Create the table
    plane_id = p.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = p.createMultiBody(baseCollisionShapeIndex=plane_id)
    pbu.set_pose(ground_id, ((0, 0, 0.01), (0, 0, 0, 1)))
    
    
    # Create the stewart platform
    taskboard_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "taskboard")
    platform_path = os.path.join(taskboard_path, "stand_for_stewart_platform_assembly_v69_vhacd.obj")
    platform = load_obj(platform_path)
    pbu.set_pose(platform, PLATFORM_POSE)

    # # Create the skateboard
    skateboard = load_obj(os.path.join(taskboard_path, "just_a_medium_cube.stl"), mesh_scale=[0.001, 0.001, 0.001])
    SKATEBOARD_POSE = ([-0.5, 0.09868, 0.09770], [0, 0, 0, 1])
    pbu.set_pose(skateboard, SKATEBOARD_POSE)

    obstacles = [ground_id, skateboard, platform]
    print("Robots: "+str([r0, r1]))
    print("Obstacles: "+str(obstacles))
    return [r0, r1], obstacles


def load_obj(stl_file_path, mass=1.0, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1], mesh_scale=[1, 1, 1]):
    """
    Loads an STL file into PyBullet as a multibody object.

    Args:
        stl_file_path (str): Path to the STL file.
        mass (float): Mass of the object.
        base_position (list): Initial position [x, y, z] of the object.
        base_orientation (list): Initial orientation [x, y, z, w] (quaternion) of the object.
        mesh_scale (list): Scale factors [x, y, z] for the mesh.

    Returns:
        int: The body ID of the created object.
    """
    # Create visual shape from the STL
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create collision shape from the STL (optional, for physics interactions)
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create the multibody object using the visual and collision shapes
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        baseOrientation=base_orientation
    )

    return body_id

# check where these are used

def get_scripted_actions(num_envs, num_robots=1, repeat=10):
    scripted_actions = [torch.tensor([[0, 0, 1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 0, -1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, -1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[-1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat
    return scripted_actions


# ---------------- EXECUTION ENVIRONMENT -----------------#
# anything exclusively used by the ARAAS digital twin environment - most setup is embedded in the araas workcell

WORKCELL_NAME="MAR_PEG_85"


# ---------------- SHARED TASK CONFIG CLASS ----------------#


@dataclass
class TaskConfig():
    PEG_ASSET_NAME: str = None
    PEG_TYPE: str = "part"
    PEG_ATTACHMENT_PRIM: str = None

    HOLE_ASSET_NAME: str = None
    HOLE_TYPE: str = "part"
    HOLE_ATTACHMENT_PRIM: str = None

    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    HOLE_T_PEG_GOAL: Tuple = None
    PEG_GOAL: Tuple = None
    HOLE_GOAL: Tuple = None

    # These are used if the relevant part of the peg/hole is separate from (???)
    # The target could be defined by pose, but this can be used if the pose doesn't matter much

    # In most cases, these can be kept at identity
    PEG_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])
    HOLE_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])

    WORLD_T_PEG_START: Tuple = None
    WORLD_T_HOLE_START: Tuple = None

    PEG_T_TIP: Tuple = None
    HOLE_T_TIP: Tuple = None

    WORLD_T_PEG_PICK: Tuple = None
    WORLD_T_HOLE_PICK: Tuple = None

    relative_goal: bool = True

    allow_peg_rotation: bool = False
    allow_hole_rotation: bool = False

    # Weights on the x, y, z, rx, ry, rz components of the error
    peg_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    # Only used if absolute pose and hole part
    hole_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    PEG_IK_WORLD_T_TOOL: Tuple = None
    HOLE_IK_WORLD_T_TOOL: Tuple = None
    
    origin_regularization: float = 0

    relative_observations: bool = False
    force_sensing: bool = False

    mask_peg: bool = False
    mask_hole: bool = False

    # Weights on the x, y, z, rx, ry, rz components of domain randomization
    randomizations: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    @property
    def PEG_T_TOOL(self):
        return pbu.multiply(self.PEG_T_TIP, pbu.invert(TOOL_T_TIP))
    
    @property
    def HOLE_T_TOOL(self):
        return pbu.multiply(self.HOLE_T_TIP, pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_TOOL_PICK_PEG(self):
        if(self.WORLD_T_PEG_PICK is None):
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_PEG_PICK, self.PEG_T_TIP), pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_TOOL_PICK_HOLE(self):
        if(self.WORLD_T_HOLE_PICK is None):
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_HOLE_PICK, self.HOLE_T_TIP), pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_PEG_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_PEG_START, self.PEG_T_TOOL)

    @property
    def WORLD_T_HOLE_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_HOLE_START, self.HOLE_T_TOOL)

    @property
    def holding_peg(self):
        return self.PEG_T_TIP is not None
    
    @property
    def holding_hole(self):
        return self.HOLE_T_TIP is not None
    
    @property
    def moving_peg(self):
        return self.holding_peg and not self.mask_peg
    
    @property
    def moving_hole(self):
        return self.holding_hole and not self.mask_hole
    
    @property
    def robot_count(self):
        return int(self.moving_peg)+int(self.moving_hole)


# ---------------- TASK CONFIG FUNCTIONS ------------------- #

# To add a new task, add any necessary constant transforms to the list at the top
# and then define the task initialization paths and transforms accordingly

def fmb_example():
    PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])

    
    WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    WORLD_T_PEG_START = pbu.multiply(([0, 0, 0.1], [0,0,0,1]), pbu.multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005",
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd"),
                      HOLE_TYPE = "peripheral",
                      HOLE_T_PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP)

def screw_in_bolt():
    WORLD_T_HOLE_START = PLATFORM_POSE
    HOLE_T_PEG_GOAL = ([[-0.00792, 0.0856, 0.41553], 
                        [0.62253, 0.33534, -0.62253, 0.33534]])
    WORLD_T_PEG_START = pbu.multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL)
    WORLD_T_PEG_GOAL = WORLD_T_PEG_START

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/bolt.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/precision_shoulder_screw/node_/mesh_",
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/platform.usd"),
                      HOLE_TYPE = "peripheral",
                      HOLE_T_PEG_GOAL = WORLD_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      PEG_T_TIP = PEG_T_TIP)

def bolt_in_strut():
    WORLD_T_PEG_START = pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT) # Strut starts standing up
    WORLD_T_HOLE_START = pbu.multiply(pbu.multiply(WORLD_T_PEG_START, STRUT_T_BOLT), ([0.005, -0.013, -0.04], [0, 0, 0, 1]))

    PEG_T_TIP = STRUT_GRASP # Strut grasp
    HOLE_T_TIP = BOLT_GRASP # Bolt grasp
    
    WORLD_T_BOLT_PICK = pbu.multiply(WORLD_T_KIT, KIT_T_BOLT_0)


    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/strut_real_shifted.usd"),
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/bolt_small.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/strut_real/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/precision_shoulder_screw/node_/mesh_",
                      HOLE_T_PEG_GOAL = pbu.invert(STRUT_T_BOLT),
                      WORLD_T_PEG_START = WORLD_T_PEG_START, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START, 
                      PEG_T_TIP = PEG_T_TIP, 
                      HOLE_T_TIP = HOLE_T_TIP, 
                      WORLD_T_HOLE_PICK = WORLD_T_BOLT_PICK,
                      EXTRA_PARTS = [os.path.join(ROOT_DIR, "taskboard/elbow_on_stand.usd")],
                      EXTRA_PARTS_STARTING_POSE=[PLATFORM_POSE],
                      EXTRA_ATTACHMENTS=[("extra0/on_drive_north", "peg/strut_real/node_/mesh_")],
                      peg_goal_weights=[1, 5, 1, 0, 0, 0],
                      randomizations=[0.0025,0.0025,0.0025,0,0,0],
                      mask_peg=True)

def elbow_in_platform():

    print("Configuring elbow in platform")
    
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.03], [0, 0, 0, 1]), pbu.multiply(PLATFORM_POSE, PLATFORM_T_ELBOW))
    WORLD_T_ELBOW = pbu.multiply(WORLD_T_KIT, KIT_T_ELBOW_0)
    # changing the peg start pose doesn't seem to change where the policy ultimately sends the elbow
    # moving the kit will have subtly impacted the grasp pose in a way that probably isn't accounted for. Hmm.

    return TaskConfig(
            PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/platform.usd"),
            HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/elbow_no_articulation_root.usd"),
            PEG_TYPE="peripheral",
            HOLE_ATTACHMENT_PRIM = "hole/on_drive_south/node_/mesh_",
            HOLE_T_PEG_GOAL=pbu.invert(PLATFORM_T_ELBOW),
            WORLD_T_HOLE_START=WORLD_T_PEG_START,
            WORLD_T_PEG_START=PLATFORM_POSE,
            WORLD_T_HOLE_PICK = WORLD_T_ELBOW, 
            HOLE_T_TIP=ELBOW_GRASP,
            randomizations=[0.005,0.005,0.005,0,0,0])


def bolt_in_platform():
    # Where the insertion starts. Not where the object starts out
    WORLD_T_HOLE_START = pbu.multiply(pbu.multiply(PLATFORM_POSE, PLATFORM_T_BOLT), ([0, 0, -0.03], [0, 0, 0, 1])) # Strut starts standing up
    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/elbow_on_stand.usd"),
                      PEG_TYPE="peripheral",
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/bolt_small.usd"),
                      HOLE_ATTACHMENT_PRIM = "hole/precision_shoulder_screw/node_/mesh_",
                      HOLE_T_PEG_GOAL = pbu.invert(PLATFORM_T_BOLT),
                      WORLD_T_PEG_START = PLATFORM_POSE, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START,
                      HOLE_T_TIP = BOLT_GRASP,
                      peg_goal_weights=[1, 1, 1, 0, 0, 0],
                      randomizations=[0.005,0.005,0.005,0,0,0])

# No longer in use and untested because I'm not sure how rotations work with the admittance controller
def combined_bolt_in_platform():

    PEG_GOAL = pbu.multiply(PLATFORM_POSE, ([[-0.00148, 0.13924, 0.23464], [-0.09318, -0.12633, 0.87414, -0.4596]]))
    HOLE_GOAL = pbu.multiply(PLATFORM_POSE, ([[-0.00792, 0.0856, 0.41553], [0.62253, 0.33534, -0.62253, 0.33534]]))

    # Where the insertion starts. Not where the object starts out
    WORLD_T_PEG_START = pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT) # Strut starts standing up
    WORLD_T_HOLE_START = pbu.multiply(HOLE_GOAL, ([0, 0, -0.08], [0, 0, 0, 1]))

    PEG_T_TIP = STRUT_GRASP # Strut grasp
    HOLE_T_TIP = BOLT_GRASP # Bolt grasp
    
    WORLD_T_BOLT_PICK = pbu.multiply(WORLD_T_KIT, KIT_T_BOLT_0)

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/strut_real.usd"),
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/bolt_small.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/strut_real/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/precision_shoulder_screw/node_/mesh_",
                      PEG_GOAL = PEG_GOAL,
                      HOLE_GOAL = HOLE_GOAL, 
                      WORLD_T_PEG_START = WORLD_T_PEG_START, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START, 
                      PEG_T_TIP = PEG_T_TIP, 
                      HOLE_T_TIP = HOLE_T_TIP, 
                      WORLD_T_HOLE_PICK = WORLD_T_BOLT_PICK,
                      EXTRA_PARTS = [os.path.join(ROOT_DIR, "taskboard/elbow_on_stand.usd")],
                      EXTRA_PARTS_STARTING_POSE=[PLATFORM_POSE],
                      EXTRA_ATTACHMENTS=[("extra0/on_drive_north", "peg/strut_real/node_/mesh_")],
                      relative_goal=False,
                      allow_peg_rotation=True,
                      peg_goal_weights=[1, 1, 1, 1, 1, 1],
                      hole_goal_weights=[5, 1, 5, 0, 0, 0])

def strut_in_elbow():
    print("Configuring strut in elbow")
    PEG_T_HOLE_GOAL = ([[0.01906, -0.01631, -0.0378], [1, 0, 0., 0.]])
    HOLE_T_PEG_GOAL = pbu.invert(PEG_T_HOLE_GOAL)

    # Where the insertion starts. Not where the object starts out
    WORLD_T_PEG_START = ([0.05, 0.125, 0.3217], [0.0000, 0.7071,  0.0000, 0.7071])
    WORLD_T_HOLE_START = pbu.multiply(WORLD_T_PEG_START, pbu.multiply(PEG_T_HOLE_GOAL, ([[0, 0, 0.045], [0., 0., 0., 1.]])))

    
    WORLD_T_STRUT = pbu.multiply(WORLD_T_KIT, KIT_T_STRUT_5)
    WORLD_T_ELBOW = pbu.multiply(WORLD_T_KIT, KIT_T_ELBOW_0)

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/strut_real.usd"),
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/elbow_no_articulation_root.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/strut_real/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/on_drive_south/node_/mesh_",
                      HOLE_T_PEG_GOAL = HOLE_T_PEG_GOAL, 
                      WORLD_T_PEG_START = WORLD_T_PEG_START, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START, 
                      PEG_T_TIP = STRUT_GRASP, 
                      HOLE_T_TIP = ELBOW_GRASP, 
                      WORLD_T_PEG_PICK = WORLD_T_STRUT, 
                      WORLD_T_HOLE_PICK = WORLD_T_ELBOW,
                      peg_goal_weights=[5, 5, 1, 0, 0, 0],
                      relative_observations=False)


def wrench_in_bearing():


    # Where the insertion starts. Not where the object starts out
    WORLD_T_PEG_START = ([0.05, 0.125, 0.3217], [0.7071, 0.00,  0.7071, 0])
    WORLD_T_HOLE_START = pbu.multiply(WORLD_T_PEG_START, WRENCH_T_BEARING_START)

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/thing2.usd"),
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/thing1.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/thing2_v2/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/thing1_v1/node_/mesh_",
                      HOLE_T_PEG_GOAL = pbu.invert(WRENCH_T_BEARING_GOAL),
                      WORLD_T_PEG_START = WORLD_T_PEG_START,
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START,
                      PEG_T_TIP = WRENCH_GRASP,
                      HOLE_T_TIP = BEARING_GRASP,
                      WORLD_T_PEG_PICK = (),
                      WORLD_T_HOLE_PICK = (),
                      peg_goal_weights=[1, 1, 1, 0.1, 0.1, 0.1],
                      allow_peg_rotation=True,
                      allow_hole_rotation=True)

def strut_in_platform():
    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT))
    bolt_start = pbu.multiply(pbu.multiply(pbu.multiply(PLATFORM_POSE, PLATFORM_T_STRUT), STRUT_T_BOLT), ([0, 0, -0.03], [0, 0, 0, 1]))
    constraint = pbu.multiply(bolt_start, pbu.multiply(BOLT_GRASP, pbu.invert(TOOL_T_TIP)))
    
    WORLD_T_STRUT = pbu.multiply(WORLD_T_KIT, KIT_T_STRUT_5)

    return TaskConfig(
            PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/strut_real.usd"),
            HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/elbow_on_stand.usd"),
            HOLE_TYPE="peripheral",
            PEG_ATTACHMENT_PRIM = "peg/strut_real/node_/mesh_",
            HOLE_IK_WORLD_T_TOOL = constraint,
            HOLE_ATTACHMENT_PRIM = None,
            HOLE_T_PEG_GOAL=PLATFORM_T_STRUT,
            WORLD_T_PEG_START=WORLD_T_PEG_START,
            WORLD_T_HOLE_START=PLATFORM_POSE,
            WORLD_T_PEG_PICK = WORLD_T_STRUT, 
            PEG_T_TIP=STRUT_GRASP,
            randomizations=[0.005,0.005,0.005,0,0,0])

def med_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_MEDGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.21818, 0.05686, 0.03195], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_medium.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_MEDGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      EXTRA_PARTS = [os.path.join(ROOT_DIR, "taskboard/gear_small.usd"), os.path.join(ROOT_DIR, "taskboard/gear_large.usd")],
                      EXTRA_PARTS_STARTING_POSE=[pbu.multiply(TASKBOARD_POSE, PLATFORM_T_SMALLGEAR), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_LARGEGEAR)],
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)

def small_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_SMALLGEAR))
    
    
    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    
    TASKBOARD_T_PICK_PEG = ([0.14292, 0.05686, -0.03561], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_small.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_20_10_B_10_Gear_20teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_SMALLGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)

def large_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.030], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_LARGEGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.2926, 0.05686, -0.03619], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_large.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_60_10_B_10_Gear_60teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_LARGEGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)

def rod_in_gear():

    # TODO
    HOLE_T_PEG_GOAL = ([[0., 0., 0.], [0., 0., 0., 1.]])

    # Where the insertion starts. Not where the object starts out
    # oh man this is way too close to the pick points! Move Y.
    WORLD_T_PEG_START = ([0.05, 0.4200, 0.4217], [0.0000, -0.7071,  0.0000, 0.7071])
    WORLD_T_HOLE_START = ([-0.052,  0.4207,  0.4357], [0.0000, 0.7071,  0.0000, 0.7071])

    # which robot is which? Erroneous offset seems to be about HOLE robot, and is ~7 cm, which is mental
    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    HOLE_T_TIP = BOLT_GRASP #pbu.invert(([0, 0, 0], [-1, 0, 0, 0]))

    #[0.21818, 0.05686, 0.03195]

    TASKBOARD_T_PICK_PEG = ([0.32418, 0.03686, -0.02195], [0.0, -0.70711, 0.70711, 0.0]) # not sure why this isn't coded in med gear setup 0.06686

    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    WORLD_T_BOLT_PICK = pbu.multiply(WORLD_T_KIT, KIT_T_BOLT_0) #BOLT_PICK_POSE

    #WORLD_T_TOOL_PICK_PEG = GEAR_PICK_POSE
    #WORLD_T_TOOL_PICK_HOLE = 
    

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/gear_medium.usd"),
                      HOLE_ASSET_NAME =  os.path.join(ROOT_DIR, "taskboard/TruckBolt.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = "hole/precision_shoulder_screw/node_/mesh_",
                      HOLE_T_PEG_GOAL = HOLE_T_PEG_GOAL, 
                      WORLD_T_PEG_START = WORLD_T_PEG_START, 
                      WORLD_T_HOLE_START = WORLD_T_HOLE_START, 
                      PEG_T_TIP = PEG_T_TIP, 
                      HOLE_T_TIP = HOLE_T_TIP, 
                      WORLD_T_PEG_PICK = WORLD_T_PEG_PICK, 
                      WORLD_T_HOLE_PICK = WORLD_T_BOLT_PICK,
                      peg_goal_weights=[4, 4, 1, 0, 0, 0])

def task_from_name(name)->TaskConfig:
    return globals()[name]()
    
