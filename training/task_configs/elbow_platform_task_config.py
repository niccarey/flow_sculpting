# task_from_name replacement: med_gear_in_taskboard
# (using UR environment with demo workcell settings/constants)
# This does not replace utils (yet), just anything configured in the task setting
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

from multi_arm_assembly.utils import solve_ik 
import multi_arm_assembly.pb_utils as pbu

import warp as wp
from pxr import Usd, UsdGeom, Gf
import trimesh
from urdfpy import URDF # hope this works, very dumb requirements list on urdfpy


# I don't really like having task config as a class, it should probably be a dict, really
# but class does let us wrap some methods that make things more modular. Ok, FINE!!

# tip == grasp point (I think?)
# peg == gear

# NOTE: now the isaac workcell layout reflects the araas layout, we shouldn't have to do insane transforms 
# on goal poses during execution - step through that and take out isaac->araas conversion constants

ROOT_DIR = Path('/home/aidanc/multi-robot-assembly/')  #Path(__file__).parent.parent.parent
#not a great way of getting the root dir!
# print(ROOT_DIR)


#------------WORKCELL CONSTANTS--------------------
# Stewart platform details:
# This is all situated relative to the robot, I think origin is still on robot
# So we still need to adjust araas world origin to match training points, probably
# however since we're trainiing from scratch: could also adjust isaac workcell poses?

# See how things boot to isaac, then experiment with adjusting world locations.
# .. robot spawn location might need changing, then.
PLATFORM_POSE = ([0.00, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]) # mounting offset - slightly above table, rotated 120 degrees around Z
# This still may not be quite right - seeing 2/3 mm offsets when elbow is enveloping peg
PLATFORM_T_ELBOW = ([0.00, 0.15, 0.168], [0, 0, 0.70711, 0.70711]) 

# this is the goal pose of the elbow IN THE PLATFORM FRAME - does it include a Z rotation? Otherwise I cannot understand the X value here
# 90 degree rotation around Z, that's it :/ WHOSE Z? I don't understand this setup at all

# Kit details: need to adjust these to match new kit position in workcell
# Decide which robot does what! Elbow = R1, strut = R0 for now
# Elbow kit sits on R1 side, strut kit sits on R0 side
WORLD_T_KIT = ([-0.005, 0.288, 0.011], [0, 0, -0.70711, 0.70711]) #Kit position in workcell
KIT_T_ELBOW_0 = ([0.09273, -0.07578, -0.01149], [0, -0.70711, 0.70711, 0]) # internal kitting transforms

#-------------GRASP CONSTANTS ---------------------
# Grasp is defined relative to insertion point

# something is ~90 degrees off - either grasp or goal point? or elbow ...

# How to grasp the elbow? V hard to trouble shoot this in isaac, as I can't load the robot to look at it without the solve.

ELBOW_T_GRASP = ([0, 0.0, 0.02], [ 0.5, -0.5, -0.5, 0.5 ]) # [0, 0.70711, 0.70711, 0]) # [-0.8525245, 0, 0,0.5226872]

#-------------ROBOT CONFIG AND POSE CONSTANTS---------------
#R0_POSE = ((0,0,0), (0,0,0,1)) # only for pseudo environment

R0_POSE = ((0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, 0.7071068, 0.7071068))
R1_POSE = ((-0.8367000122070312, 0.6095999755859375, 0.02250), (0, 0, -0.7071068, 0.7071068))

RQ85_TCP_OFFSET = ([0,0,0.285],[0,0,0,1]) # replaces TOOL_T_TIP. 


#-------------RELATIVE POSE CONSTANTS--------------
# Calculated constants
# DEBUG: put elbow in wrong position to try and force a solve
#print(pbu.multiply(PLATFORM_POSE, PLATFORM_T_ELBOW))

# There's an issue with origin offset on elbow which is poking its head up

WORLD_T_ELBOW_START = pbu.multiply(([0, 0, 0.03], [0,0,0,1]), pbu.multiply(PLATFORM_POSE, PLATFORM_T_ELBOW))
#print(WORLD_T_ELBOW_START)

# Static identifiers
TOOL_NAME = "tool0"

# FUNCTIONAL CONFIG DEFINITION (see TaskConfig for what we can actually extract, and environment for what we actually need)
# Trying to switch out peg/hole (confusing, changes definition!) to fixed/moving, to clarify role in scene
# means we have to be careful about which robot is in control, but long run should be easier
# Keeping plug/socket terminology in mesh loading and sdf counts, because it matters which is on the outside.
# swap plug/socket between fixed/moving as needed when calling.


# TODO: make usds of the elbow interior and platform peg, add asset and prim paths here.
# TODO: check hole and pin radii in CAD (invent a specious tolerancing if necessary)

@dataclass
class ElbowPlatformConfig():

    # baseline task settings:
    # Still need to clean up nomenclature - let's use FIXED and MOVING for single robot tasks
    # part/peripheral not relevant for training, and we leave the araas set up to fend for itself mostly
    robot_count = 1
    #moving_peg = True 
    #moving_hole = False 
    moving_robot = "R0" #probably R1, I think? (TODO: shift pickup in araas) - also, depends which peg we pick.

    relative_goal = True
    FIXED_ASSET_NAME: str = os.path.join(ROOT_DIR, "stewart_platform/platform.usd")
    FIXED_TYPE: str = "peripheral"
    #FIXED_ATTACHMENT_PRIM: str = "peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_"

    MOVING_ASSET_NAME: str = os.path.join(ROOT_DIR, "stewart_platform/fixed_elbow.usd") #elbow_no_articulation_root.usd")
    MOVING_ATTACHMENT_PRIM: str = "moving/fixed_elbow/Elbow_v17/Body81_01"

    MOVING_TYPE: str = "part"
    R0_POSE = R0_POSE
    R1_POSE = R1_POSE

    #for calculating SDF rewards:

    fixed_sdf_asset =os.path.join(ROOT_DIR,  "stewart_platform/lone_peg.usd")
    #moving_sdf_asset =os.path.join(ROOT_DIR,  "stewart_platform/elbow_sdf_asset.usd")
    #moving_prim_path = "/World/on_drive_south/node_/mesh_"#f"/World/envs/env_{0}/peg"  #"

    moving_sdf_asset = os.path.join(ROOT_DIR,"stewart_platform/fixed_elbow.usd")
    moving_prim_path = "/World/fixed_elbow/Elbow_v17/Body81_01"
    fixed_prim_path = "/World/platform_peg/platform_peg/Body1" #f"/World/envs/env_{0}/hole"# 
    moving_xform_path = "/World/fixed_elbow"#f"/World/envs/env_{0}/peg"  #"
    fixed_xform_path = "/World/platform_peg"


    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    # (probably don't need for training purposes, but do add to araas workcell)
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    # control and action settings
    relative_observations: bool = False
    force_sensing: bool = False

    allow_moving_rotation: bool = False
    allow_fixed_rotation: bool = False

    origin_regularization: float = 0 # I don't think this is ever used?

    # Weights on the x, y, z, rx, ry, rz components of the error
    # is weighting off?

    # for all gears we just use default weights ... but actually, we might want to weight X/Y higher?
    # doing a rough search - seem to get best results with 0.7 (best still not being amazing)
    moving_goal_weights: List[float] = field(default_factory=lambda: [1.5, 1.5, 0.7, 0, 0, 0])

    # Don't really use this,doesn't make a lot of sense in the context of a single robot
    fixed_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    # Weights on the x, y, z, rx, ry, rz components of domain randomization (no longer used, I think?)
    randomizations: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    # sdf reward function parameters
    sdf_num_samples = 200
    sdf_weighting = 1 # not sure yet
    # max we can get from sdf looks to be around 5/6. We CAN'T get a completion bonus without sdf  
    engagement_weighting = 8  # 10 might actually be too big, weirdly - we don't reinforce good but not perfect behaviours enough?
    engagement_threshold = 0.001 # 0.5mm? too low?
    overlap_weighting = 0.1;
    # can also have an engagement bonus (fixed reward add)

    # non-default global pose constants
    # This looks like hot garbage!
    WORLD_T_MOVING_START = WORLD_T_ELBOW_START
    #print("Sanity check for peg placement")
    #print(WORLD_T_MOVING_START)

    WORLD_T_FIXED_START = PLATFORM_POSE # we're not picking up the platform, so this is a spawn point only? 

    MOVING_T_TIP = ELBOW_T_GRASP
    MOVING_T_TOOL = pbu.multiply(ELBOW_T_GRASP, pbu.invert(RQ85_TCP_OFFSET))

    MOVING_T_GOAL = PLATFORM_T_ELBOW # ah. Offset between (taskboard base) and (gear final resting pose). Would be zero if these aligned.

    WORLD_T_MOVE_TOOL_START = pbu.multiply(WORLD_T_MOVING_START, MOVING_T_TOOL)

    # Need to get this from CAD!
    socket_radius = 0.0075
    plug_radius = 0.005

    # OFFSETS AND TRANSFORMS FOR SAMPLING SDFs:
    # need to recalculate these - especially look out for elbow origin to hole base, and platform origin to pin base.

    # elbow base should be hole, elbow top should be extent of hole (not actually relevant, I think?)
    elbow_base_T_elbow_top = ([0,0,0.03], [0,0,0,1])

    # elbow origin may not be at base. Transform between elbow origin frame and hole frame
    # fusion origin is now at hole frame, but needs to be flipped
    elbow_orig_T_elbow_base = ([0,0,0], [1,0,0,0]) # It's possible this needs adjusting, because of Fusion origin. (we could also re-export :/)
    
    # assuming Z axis is aligned correctly on peg base pose, then
    # transform between pin base and top of pin (v small pin)
    pin_base_T_pin_tip = ([0,0,0.008], [0,0,0,1])# pin is the insertion-relevant subpart. Should be a small offset purely in Z.
    platform_origin_T_pin_base = PLATFORM_T_ELBOW  # This should be the origin point at which we spawn the target pin. 
    #Pin is actually quite long, but spawn origin is at centre, so base_t_tip offset is not large.

    # use this to calculate conical attraction field
    # if we want to use this directly instead of calculating it every time.
    platform_orig_T_pin_tip = pbu.multiply(platform_origin_T_pin_base, pin_base_T_pin_tip) 

    # Generalize names for use in single-robot environment:
    # These names are very confusing, actually.
    moving_base_T_top = elbow_base_T_elbow_top # Not used for now

    moving_orig_T_base = elbow_orig_T_elbow_base # Origin to base offset for elbow part
    fixed_base_T_top = pin_base_T_pin_tip # delta between bottom of pin and top of pin
    fixed_origin_T_base = platform_origin_T_pin_base # platform origin to bottom of pin
    fixed_origin_T_top = platform_orig_T_pin_tip # platform origin to top of pin



def get_elbow_config():
   return ElbowPlatformConfig()



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




# every config file should have an environment setup function with only the things we care about.
# for gear in taskboard: one robot, maybe the task board?

# client is by default pybullet, not sure why. Just lighter than physx?
# why is this even happening outside isaac?

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
    #print("sending through bc client")
    #print(client)
    print("Environment set, attempting to solve")
    ik_solutions = solve_ik(start_qs, target_poses, robots, obstacles, tool_name=TOOL_NAME, ee_collisions=True, step_through=False, client=client)

    # may need to force this to float?

    if len(ik_solutions) > 0:
        return ik_solutions
    else:
        print("Unable to solve requested target")