# bolt_eye_task_config
# trying to remove reliance on unsorted utility files/functions

import pybullet as p
import numpy as np 
import time 
import os
import math
import pathlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import pybullet_utils.bullet_client as bc

from multi_arm_assembly.training.robot_kinematic_configs.ur_kinematic_config import solve_ik 

import pb_utils as pbu


import warp as wp
from pxr import Usd, UsdGeom, Gf
import trimesh
from urdfpy import URDF 


ROOT_DIR = Path('/home/aidanc/multi-robot-assembly/')

# Locate assets
#USD_DIR = Path("/home/aidanc/stewart_platform/") # point to folder that contains usd files
URDF_DESC = Path("/home/aidanc/multi-robot-assembly/multi_arm_assembly/ur_description/urdf/")

# END-EFFECTOR TOOLING
TOOL_NAME = "tool0" # replace with screwdriver USD
RQ85_TCP_OFFSET = ([0,0,0.295],[0,0,0,1]) # replace with offset distance for screwdriver tip

# When we swap the gripper, can also take out knuckle joint (and adjust environment accordingly)
ROBOT_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "left_outer_knuckle_joint",
]

# Static identifiers
TOOL_NAME = "tool0"


# Default UR joint configuration (not necessarily the insertion starting pose for the robot)
#this should be robot config, not robot pose
#[-0.9920, -1.10733, -2.4733, -1.1288, 1.5637, 0.01950, 0.0] #[-0.5144, -1.643, -2.105, -0.95356, 1.561, -0.5144, 0.0]

# use this once on every setup 
# includes finger joint (if used) 
R0_POSE = ((0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, 0.7071068, 0.7071068)) 


# TASK-SPECIFIC ENVIRONMENT CONSTANTS - there's a lot of these, so wrapping in a dataclass for accessibility
# will have to reconfigure this (and the USD, actually) for a moving eye part


# HERE WE GO!!!

@dataclass
class BoltEyeConfig():

    # residual info from GoFlow training setup, should be able to eliminate these if we write an environment-specific agent
    robot_count = 1
    moving_robot = "R0" # can't remember how much we use this now
    relative_goal = True
    R0_POSE = R0_POSE

    FixedPlatformAsset: str = os.path.join(ROOT_DIR, "stewart_platform/bolt_insert_on_platform.usd")
    FIXED_TYPE: str = "peripheral" # shoudl also be able to get rid of these

    BoltAsset: str = os.path.join(ROOT_DIR, "stewart_platform/bolt.usd")
    MOVING_TYPE: str = "part"

    # primitives used to create rigid (or other) attachments - path can be arbitrary, is sent to USD on load
    # I don't THINK we need the fixed prim for this configuration
    PlatformPrim: str  = "fixed/ball_socket/ball_socket/Body1" 
    BoltPrim: str = "moving/precision_shoulder_screw/node_/mesh_"

    BallJointPath: str = "/World/ball_socket"


    # SDF information
    fixed_sdf_asset = os.path.join(ROOT_DIR, "stewart_platform/ball_socket_alone.usd")
    moving_sdf_asset =os.path.join(ROOT_DIR, "stewart_platform/bolt.usd")

    # These have to be in accordance with the USD file structure
    moving_prim_path = "/World/precision_shoulder_screw/node_/mesh_"
    moving_xform_path = "/World/precision_shoulder_screw"

    fixed_prim_path = "/World/ball_socket/ball_socket/Body1"  
    fixed_xform_path = "/World/ball_socket"

    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    # the way this is structured, it seems like the attachment meshes must already be in the usd structure?

    # no longer doing it this way, I think? But need to make sure stand and ball socket are added    
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    # control and action settings
    relative_observations: bool = False
    force_sensing: bool = False

    allow_moving_rotation: bool = False # MAY WANT TO SET TO TRUE - hmm, not sure this is significantly impacting success rate
    allow_fixed_rotation: bool = False

    origin_regularization: float = 0 # I don't think this is ever used?

    # doing a rough search - seem to get best results with 0.7 (best still not being amazing)
    # if we allow moving rotation, need to also set non-zero motion weights
    # Check whether these are in world frame or grasp frame (I think world frame) - in which case Y is the insertion axis
    moving_goal_weights: List[float] = field(default_factory=lambda: [1.2, 0.7, 1.2, 0,0,0])

    # sdf reward function parameters
    sdf_num_samples = 200
    sdf_weighting = 1 # not sure yet

    # No longer using conic overlap - too hard to code / too heuristic to pull from mesh data (without a widget)

    # max we can get from sdf looks to be around 5/6. We CAN'T get a completion bonus without sdf  
    engagement_weighting = 8  # 10 might actually be too big, weirdly - we don't reinforce good but not perfect behaviours enough?
    engagement_threshold = 0.001 # 0.5mm? too low? # probably not using this either?
    overlap_weighting = 0.1; # don't use

    # Changing to a setup specific agent training environment, trying to be general just creates problems. Renaming everything!
    PlatformOrigin = ([[0.005, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]])
    BallOffset = ([0.0505, 0.064, 0.401], [0.7071068, 0, 0,0.7071068])
    BallOrigin = pbu.multiply(PlatformOrigin, BallOffset) 

    #PlatformStrutOffset = ([[0.0002, 0.15019, 0.23461], [0, 0, -0.88691, 0.46195]])
    # define final offset pose from CAD - ball origin to bolt origin
    #StrutBoltOffset = ([[0, 0, 0.20243], [0, 0.70711, 0, 0.70711]]) # distance between strut and bolt after insertion - goal pose
    # assume we can get fairly close to the strut to start the insertion - initialise at 3cm (plus/minus randomization)
    #BoltOrigin = pbu.multiply(pbu.multiply(StrutOrigin, StrutBoltOffset), ([0, 0, -0.03], [0, 0, 0, 1])) 

    # or define experimentally from ISAAC:
    # ah, yeah, there's some weird origin transform issues here - oh, I'm multiplying by the wrong thing, that's why
    PlatformBoltOffset = ([[0.0505, 0.083, 0.42],  [0.7071068, 0, 0,0.7071068]])
    # Check this - multiplication order may be wrong ... oh, and I was wrong about the Z axis of the ball joint.
    # Construct relative to platform, not ball joint

    BoltGoalPose = pbu.multiply(PlatformOrigin, PlatformBoltOffset)
    BoltStartPose = pbu.multiply(BoltGoalPose, ([0, 0, -0.024], [0, 0, 0, 1])) 
    # 1.5 cm along bolt negative Z, I think? But need to be in world coords

    # grasp offsets: (currently hardcoded, ideally pull in from perception+grasp module)
    BoltGrasp = ([0, 0, -0.02], [0, 0, 0, 1]) # Z rotation is irrelevant for symmetric part, just move slightly away from the origin 
    BoltToolOffset = pbu.multiply(BoltGrasp, pbu.invert(RQ85_TCP_OFFSET))

    # moving tool transforms:
    MOVING_T_TIP = BoltGrasp
    MOVING_T_TOOL = BoltToolOffset
    MOVING_T_GOAL = pbu.multiply(pbu.invert(BoltGoalPose), BallOrigin ) # Offset between (ball origin) and (bolt origin). 
    # Would be zero if these aligned, BUT THEY DON'T. 
    # ... since we know the goal world coordinates for the bolt, can just use those?
    # ... something is off though
    # print(MOVING_T_GOAL) 

    # Bolt goal pose is bolt in world frame, we are getting ball origin in bolt goal frame - ie desired offset of ball origin when we land in goal pose
    # Rotation seems right, translation seems very funky - 60mm Z, 35mm -Y 
    WORLD_T_MOVE_TOOL_START = pbu.multiply(BoltStartPose, MOVING_T_TOOL)

    # Not using conic reward - MAY want to use engagement reward
    moving_orig_T_tip = ([0,0, 0.017], [0,0,0,1]) # from CAD
    fixed_orig_T_tip = ([0,0.007, 0],[0.7071068, 0, 0,0.7071068])

    engagement_threshold = 0.024 # could get this from tip/origin configuration but let's just add it for now

def get_bolt_eye_config():
   return BoltEyeConfig()



@wp.kernel
def transform_points(
    src: wp.array(dtype=wp.vec3), dest:wp.array(dtype=wp.vec3), xform: wp.transform
    ):

    tid = wp.tid()
    p = src[tid]
    m = wp.transform_point(xform, p)
    dest[tid] = m



@wp.kernel(enable_backward=False)
def compute_tri_areas(
    points: wp.array(dtype=wp.vec3),
    face_vertex_indices: wp.array(dtype=wp.int32),
    out_tri_areas: wp.array(dtype=wp.float32),
    out_total_area: wp.array(dtype=wp.float32),
):
    tri = wp.tid()

    # Retrieve the indices of the three vertices that form the current triangle.
    vtx_0 = face_vertex_indices[tri * 3]
    vtx_1 = face_vertex_indices[tri * 3 + 1]
    vtx_2 = face_vertex_indices[tri * 3 + 2]

    # Retrieve their 3D position.
    pt_0 = points[vtx_0]
    pt_1 = points[vtx_1]
    pt_2 = points[vtx_2]

    # Calculate the cross product of two edges of the triangle,
    # which gives a vector whose magnitude is twice the area of the triangle.
    cross = wp.cross((pt_1 - pt_0), (pt_2 - pt_0))
    area = wp.length(cross) * 0.5

    # Store the result.
    out_tri_areas[tri] = area
    wp.atomic_add(out_total_area, 0, area)


@wp.kernel(enable_backward=False)
def compute_probability_distribution(
    tri_areas: wp.array(dtype=wp.float32),
    total_area: wp.array(dtype=wp.float32),
    out_probabilities: wp.array(dtype=wp.float32),
):
    tri = wp.tid()

    # Calculate the probability of selecting this triangle,
    # which is proportional to the triangle's area relative to total mesh area.
    out_probabilities[tri] = tri_areas[tri] / total_area[0]


@wp.kernel(enable_backward=False)
def accumulate_cdf(
    tri_count: wp.int32,
    out_cdf: wp.array(dtype=wp.float32),
):
    # Transform probability values into a Cumulative Distribution Function (CDF).
    for tri in range(1, tri_count):
        out_cdf[tri] += out_cdf[tri - 1]


@wp.kernel(enable_backward=False)
def sample_mesh(
    mesh: wp.uint64,
    cdf: wp.array(dtype=wp.float32),
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    rng = wp.rand_init(42, tid)

    # Sample the triangle index using the CDF.
    sample = wp.randf(rng)
    tri = wp.lower_bound(cdf, sample)

    # Sample the location in that triangle using random barycentric cordinates.
    ru = wp.randf(rng)
    rv = wp.randf(rng)
    tri_u = 1.0 - wp.sqrt(ru)
    tri_v = wp.sqrt(ru) * (1.0 - rv)
    pos = wp.mesh_eval_position(mesh, tri, tri_u, tri_v)

    # Store the result.
    out_points[tid] = pos



def load_asset_mesh(asset_path, prim_path, xform_path, sample_points, num_samples, device):
    usd_stage = Usd.Stage.Open(asset_path)
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim_path))

    # We are applying (some transform drawn from the parent) to these smaller parts, so we 
    # want the initial orientation of the smaller parts to align with the parent.

    xform = UsdGeom.Xformable(usd_stage.GetPrimAtPath(xform_path))

    part_descr = usd_stage.GetPrimAtPath(xform_path)

    local_translation = part_descr.GetAttribute("xformOp:translate").Get()
    local_rotation = part_descr.GetAttribute("xformOp:orient").Get()
    vec_scale = part_descr.GetAttribute("xformOp:scale").Get()
    local_scale = vec_scale[0]

    # Can I get part bounds in world, for debugging?
    # Extent attribute is in mm, or pre-scaling constants, which is mental

    # need to extract data. Clunky! Assume WXYZ is standard across both. Why can't we just get the data as an array??
    mesh_rotation = np.array([local_rotation.GetImaginary()[0], local_rotation.GetImaginary()[1], local_rotation.GetImaginary()[2], local_rotation.GetReal()])
    mesh_transform = wp.transform(np.array(local_translation), mesh_rotation)

    print("Checking transform on load")
    print(local_translation)
    print(mesh_rotation)

    #urdf = URDF.load(urdf_path)
    #mesh = urdf.links[0].collision_mesh

    wp_mesh = wp.Mesh(
        points=wp.array(usd_geom.GetPointsAttr().Get()*local_scale, dtype=wp.vec3),
        indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

    wp.launch(
        kernel=transform_points,
        dim = len(wp_mesh.points),
        inputs = [wp_mesh.points, wp_mesh.points, mesh_transform],
        device = device,
        )

    wp_mesh.refit()
    tri_count = len(usd_geom.GetFaceVertexIndicesAttr().Get()) // 3

    if sample_points:
        # sampling reduction - ugh, trimesh expects a triangle mesh, which is not easily available in wp_mesh 
        #sampled_points, _ = trimesh.sample.sample_surface_even(, num_samples)
        #wp_mesh_sampled_points = wp.array(sampled_points, dtype=wp.vec3, device=device)

        # Compute the area of each triangle and the total area of the mesh.
        tri_areas = wp.empty(shape=(tri_count,), dtype=wp.float32)
        total_area = wp.zeros(shape=(1,), dtype=wp.float32)

        wp.launch(
            compute_tri_areas,
            dim=tri_areas.shape,
            inputs=(
                wp_mesh.points,
                wp_mesh.indices,
            ),
            outputs=(
                tri_areas,
                total_area,
            ),
        )

        # Build a Cumulative Distribution Function (CDF) where the probability
        # of sampling a given triangle is proportional to its area.
        cdf = wp.empty(shape=(tri_count,), dtype=wp.float32)

        wp.launch(
            compute_probability_distribution,
            dim=cdf.shape,
            inputs=(
                tri_areas,
                total_area,
            ),
            outputs=(cdf,),
        )
        wp.launch(
            accumulate_cdf,
            dim=(1,),
            inputs=(tri_count,),
            outputs=(cdf,),
        )

        wp_sampled_points = wp.empty(shape=(num_samples,), dtype=wp.vec3)

        wp.launch(
            sample_mesh,
            dim=wp_sampled_points.shape,
                inputs=(
                    wp_mesh.id,
                    cdf,
                ),
                outputs=(wp_sampled_points,),
            )
        return wp_mesh, wp_sampled_points

    else:
        return wp_mesh


def load_asset_meshes_in_warp(
    moving_asset_path,
    moving_prim_path,
    moving_xform_path,
    fixed_asset_path,
    fixed_prim_path,
    fixed_xform_path,
    fixed_pose_pos,
    fixed_pose_quat,
    num_samples,
    device):
    
    # changing to a moving/fixed nomenclature
    # xform_path encodes the base transform applied to align the sdf mesh with the world frame.
    # (why does fixed pose need to be explicitly entered here? ah, presumably incorporates environment offset)

    # Debugging sdf calculations: mesh points should be in world coordinates. 
    # GUT INSTINCT: scaling is fucked

    moving_meshes, fixed_meshes, moving_meshes_sampled_points = [], [], []

    moving_mesh, sampled_points = load_asset_mesh(
        asset_path = moving_asset_path,
        prim_path = moving_prim_path,
        xform_path = moving_xform_path,
        sample_points = True,
        num_samples = num_samples,
        device = device,
        )
    moving_meshes.append(moving_mesh)
    moving_meshes_sampled_points.append(sampled_points)

    # Not entirely sure what the point of generating a new mesh is, since we don't change it? 
    # This is the IndustREAL use, and it may be to do with parallelization, actually
    # If we can parallelize the calling function, then this makes sense.



    # I think the peg location is off, even though it spawned at the right place?
    # WARP USES XYZW
    # applying transforms in the wrong order?
    
    fixed_mesh_load = load_asset_mesh(
            asset_path = fixed_asset_path,
            prim_path = fixed_prim_path,
            xform_path = fixed_xform_path,
            sample_points = False,
            num_samples = -1,
            device = device,
            )
    mesh_points = wp.clone(fixed_mesh_load.points)
    mesh_indices = wp.clone(fixed_mesh_load.indices)

    fixed_mesh = wp.Mesh(points = mesh_points, indices = mesh_indices)


    fixed_transform = wp.transform(fixed_pose_pos, fixed_pose_quat)

    wp.launch(
        kernel=transform_points,
        dim = len(fixed_mesh.points),
        inputs = [fixed_mesh.points, fixed_mesh.points, fixed_transform],
        device = device,
        )

    fixed_mesh.refit()

   
    # The socket should now be lined up with the overall USD world mesh 
    # SO: when we load the environment USD, 
    fixed_meshes.append(fixed_mesh)

    return moving_meshes, moving_meshes_sampled_points, fixed_meshes


def setup_environment(client, robot_name):
    if robot_name == "R0":
        robot = client.loadURDF("/home/aidanc/multi-robot-assembly/multi_arm_assembly/ur_description/urdf/ur10e.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
    else:
        robot = client.loadURDF("/home/aidanc/multi-robot-assembly/multi_arm_assembly/ur_description/urdf/ur10e.urdf", R1_POSE[0], R1_POSE[1], useFixedBase=True)
    
    # Create the table
    plane_id = client.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = client.createMultiBody(baseCollisionShapeIndex=plane_id)
    pbu.set_pose(ground_id, ((0, 0, -0.01), (0, 0, 0, 1)), client=client)

    # scene obstacles: since we are not using this utility to perform large motions or motion plan at all,
    # the obstacle list is left lean. See older version for taskboard/stewart/skateboard truck import, if needed.

    # should be able to move this into a task-specific config script eventually, but it is called by the IK solver I think.
    obstacles = [ground_id]
    print("Obstacles: "+str(obstacles))

    return [robot], obstacles

# wrapper passes through functional ik solver params, and performs any initialization or configuration unique to the task

def ik_solve_wrapper( start_qs, robot_name, target_poses, tool_name=TOOL_NAME, ee_collisions=True, step_through=False, client=None):
    # nominally we could change the tool in the environment without breaking this, but it might be brittle.
    # maybe tool (ie end effector) should be part of the task config?

    if(client is None):
        client = bc.BulletClient(connection_mode=p.GUI if step_through else p.DIRECT)

    robots, obstacles = setup_environment(client=client, robot_name=robot_name)

    print("Environment set, attempting to solve")
    ik_solutions = solve_ik(start_qs, target_poses, robots, obstacles, tool_name=TOOL_NAME, ee_collisions=True, step_through=False, client=client)

    # may need to force this to float?

    if len(ik_solutions) > 0:
        return ik_solutions
    else:
        print("Unable to solve requested target")
