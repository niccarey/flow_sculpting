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
import pb_utils as pbu

import warp as wp
from pxr import Usd, UsdGeom, Gf
import trimesh
from urdfpy import URDF # hope this works, very dumb requirements list on urdfpy


# I don't really like having task config as a class, it should probably be a dict, really
# but class does let us wrap some methods that make things more modular. Ok, FINE!!

# tip == grasp point (I think?)
# peg == gear

ROOT_DIR = Path(__file__).parent.parent.parent.parent #not a great way of getting the root dir!

# constants from workcell (update these to match demo cell)
# PLATFORM_POSE = ([[0.005, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]]) #this is stewart platform, I think? 

# This should be from the taskboard origin to the base of the insertion pegs.
# If we multiply these, do they cancel out? 
TASKBOARD_T_MEDGEAR = ([0.2172, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17]) 
TASKBOARD_T_SMALLGEAR =  ([0.1872, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
TASKBOARD_T_LARGEGEAR = ([0.2672, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])

TASKBOARD_POSE = ([0.150, 0.315, -0.034], [0.7071068, 0, 0, 0.7071068]) # NIST taskboard location - quat xyzw, position ok

# pick points aren't super relevant for training, can take them out
TASKBOARD_T_PICK_GEAR = ([0.21818, 0.05686, 0.03195], [-0.0, -0.70711, 0.70711, 0.0])
WORLD_T_GEAR_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_GEAR) # could also get this from gear pose

# constants from robot and gripper (check R0_POSE)
R0_POSE = ((0,0,0), (0,0,0,1))
GEAR_T_GRASP = ([0, 0, 0], [0, 0, 0, 1]) 

RQ85_TCP_OFFSET = ([0,0,0.3],[0,0,0,1]) # replaces TOOL_T_TIP. 

# 0.3m Z value pulled from araas config. 
# are we seeing a finger-->robot IK flip, maybe?

# Calculated constants
WORLD_T_GEAR_START = pbu.multiply(([0, 0.00, 0.05], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_MEDGEAR))


# Static identifiers
TOOL_NAME = "tool0"

# FUNCTIONAL CONFIG DEFINITION (see TaskConfig for what we can actually extract, and environment for what we actually need)
# I would like to move away from "peg" and "hole" terminology, it can be very confusing ...


@dataclass
class MedGearConfig():

    # baseline task settings:
    robot_count = 1
    moving_robot = "R0"
    #oving_peg = True 
    #oving_hole = False 
    relative_goal = True
    MOVE_ASSET_NAME: str = os.path.join(ROOT_DIR, "taskboard/gear_medium.usd")
    MOVE_TYPE: str = "part"
    MOVE_ATTACHMENT_PRIM: str = "peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_"
    FIXED_ASSET_NAME: str = os.path.join(ROOT_DIR, "taskboard/taskboard.usd")
    FIXED_TYPE: str = "peripheral"
    FIXED_ATTACHMENT_PRIM: str = None

    fixed_sdf_asset =os.path.join(ROOT_DIR,  "taskboard/gearshaft.usd")
    moving_sdf_asset =os.path.join(ROOT_DIR,  "taskboard/gear_medium_hollow.usd")

    moving_prim_path = "/World/gear_shaft_hollow/gear_shaft_hollow/Body1"#f"/World/envs/env_{0}/peg"  #"
    fixed_prim_path = "/World/gear_tapped_shaft/gear_tapped_shaft/Body1" #f"/World/envs/env_{0}/hole"# 
    moving_xform_path = "/World/gear_shaft_hollow"#f"/World/envs/env_{0}/peg"  #"
    fixed_xform_path = "/World/gear_tapped_shaft"


    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    # (this task has no extra parts)
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    # control and action settings
    relative_observations: bool = False
    force_sensing: bool = False

    allow_peg_rotation: bool = False
    allow_hole_rotation: bool = False

    origin_regularization: float = 0 # I don't think this is ever used?

    # Weights on the x, y, z, rx, ry, rz components of the error
    # is weighting off?

    # for all gears we just use default weights ... but actually, we might want to weight X/Y higher?
    # doing a rough search - seem to get best results with 0.7 (best still not being amazing)
    peg_goal_weights: List[float] = field(default_factory=lambda: [3, 3, 1, 0, 0, 0])

    # Only used if absolute pose and hole part
    hole_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    # Weights on the x, y, z, rx, ry, rz components of domain randomization (no longer used, I think?)
    randomizations: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    # sdf reward function parameters
    sdf_num_samples = 200
    sdf_weighting = 1 # not sure yet
    # max we can get from sdf looks to be around 5/6. We CAN'T get a completion bonus without sdf  
    engagement_weighting = 7  # 10 might actually be too big, weirdly - we don't reinforce good but not perfect behaviours enough?
    engagement_threshold = 0.0005 # 0.5mm? too low?
    overlap_weighting = 0.1;
    # can also have an engagement bonus (fixed reward add)

    # non-default global pose constants
    WORLD_T_MOVE_START = WORLD_T_GEAR_START
    WORLD_T_FIXED_START = TASKBOARD_POSE 

    PEG_T_TIP = GEAR_T_GRASP
    MOVING_T_TOOL = pbu.multiply(GEAR_T_GRASP, pbu.invert(RQ85_TCP_OFFSET))

    HOLE_T_PEG_GOAL = TASKBOARD_T_MEDGEAR # ah. Offset between (taskboard base) and (gear final resting pose). Would be zero if these aligned.

    WORLD_T_PEG_PICK = WORLD_T_GEAR_PICK # don't actually use this
    WORLD_T_PEG_TOOL_START = pbu.multiply(WORLD_T_FIXED_START, MOVING_T_TOOL)

    # Need to get this from CAD!
    hole_radius = 0.007
    peg_radius = 0.005

    gear_hole_base_T_gear_hole_top = ([0,0,0.02], [0,0,0,1])
    gear_orig_T_gear_hole_base = ([0,0,0], [0,0,0,1]) # should be aligned
    # assuming Z axis is aligned correctly on peg base pose, then
    peg_base_T_peg_tip = ([0,0,0.02], [0,0,0,1])
    board_origin_T_peg_base = ([0.2172, 0.0617, 0.2647], [-0.7071068, 0, 0, 0.7071068]) 
    #TASKBOARD_T_MEDGEAR # GOT IT! There's a 180 flip here for the gear origin, which we DON'T want, because the shaft is aligned with the board.

    board_orig_T_peg_tip = pbu.multiply(board_origin_T_peg_base,peg_base_T_peg_tip) # if we want to use this directly instead of calculating it every time.
 
    # default global pose constants (most of these aren't used in any config, and definitely not this one)

    PEG_GOAL: Tuple = None
    HOLE_GOAL: Tuple = None

    PEG_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])
    HOLE_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])

    WORLD_T_HOLE_PICK: Tuple = None
    PEG_IK_WORLD_T_TOOL: Tuple = None
    HOLE_IK_WORLD_T_TOOL: Tuple = None

    # don't use this
    WORLD_T_TOOL_PICK_PEG = pbu.multiply(pbu.multiply(WORLD_T_PEG_PICK, PEG_T_TIP), pbu.invert(RQ85_TCP_OFFSET))
    

def get_med_gear_config():
   return MedGearConfig()



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

    #print("Mesh relative transform")
    #print(local_rotation)
    #print(local_translation)

    # need to extract data. Clunky! Assume WXYZ is standard across both. Why can't we just get the data as an array??
    mesh_rotation = np.array([local_rotation.GetImaginary()[0], local_rotation.GetImaginary()[1], local_rotation.GetImaginary()[2], local_rotation.GetReal()])
    mesh_transform = wp.transform(np.array(local_translation), mesh_rotation)

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
    plug_asset_path,
    plug_prim_path,
    plug_xform_path,
    socket_asset_path,
    socket_prim_path,
    socket_xform_path,
    socket_pose_pos,
    socket_pose_quat,
    num_samples,
    device):
    
    plug_meshes, socket_meshes, plug_meshes_sampled_points = [], [], []
    
    plug_mesh, sampled_points = load_asset_mesh(
        asset_path = plug_asset_path,
        prim_path = plug_prim_path,
        xform_path = plug_xform_path,
        sample_points = True,
        num_samples = num_samples,
        device = device,
        )
    plug_meshes.append(plug_mesh)
    plug_meshes_sampled_points.append(sampled_points)

    # Not entirely sure what the point of generating a new mesh is, but anyway

    socket_mesh_load = load_asset_mesh(
            asset_path = socket_asset_path,
            prim_path = socket_prim_path,
            xform_path = socket_xform_path,
            sample_points = False,
            num_samples = -1,
            device = device,
            )
    mesh_points = wp.clone(socket_mesh_load.points)
    mesh_indices = wp.clone(socket_mesh_load.indices)

    socket_mesh = wp.Mesh(points = mesh_points, indices = mesh_indices)

    # Very confused about how this transform is being applied. 
    # I don't understand how we arrive at this result with this rotation
    # WARP USES XYZW
    socket_transform = wp.transform(socket_pose_pos, socket_pose_quat)

    wp.launch(
        kernel=transform_points,
        dim = len(socket_mesh.points),
        inputs = [socket_mesh.points, socket_mesh.points, socket_transform],
        device = device,
        )

    socket_mesh.refit()
   
    # The socket should now be lined up with the overall USD world mesh - which it is not!
    # SO: when we load the environment USD, 

    socket_meshes.append(socket_mesh)

    return plug_meshes, plug_meshes_sampled_points, socket_meshes




# every config file should have an environment setup function with only the things we care about.
# for gear in taskboard: one robot, maybe the task board?

# client is by default pybullet, not sure why. Just lighter than physx?
# why is this even happening outside isaac?

def setup_environment(client):
    r0 = client.loadURDF("/home/aidanc/multi-robot-assembly/multi_arm_assembly/ur_description/urdf/ur10e.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)
    
    # Create the table
    plane_id = client.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = client.createMultiBody(baseCollisionShapeIndex=plane_id)
    pbu.set_pose(ground_id, ((0, 0, -0.01), (0, 0, 0, 1)), client=client)

    # scene obstacles: since we are not using this utility to perform large motions or motion plan at all,
    # the obstacle list is left lean. See older version for taskboard/stewart/skateboard truck import, if needed.

    # should be able to move this into a task-specific config script eventually, but it is called by the IK solver I think.
    obstacles = [ground_id]
    print("Obstacles: "+str(obstacles))

    return [r0], obstacles

# wrapper passes through functional ik solver params, and performs any initialization or configuration unique to the task

def ik_solve_wrapper(start_qs, target_poses, tool_name=TOOL_NAME, ee_collisions=True, step_through=False, client=None):
    # nominally we could change the tool in the environment without breaking this, but it might be brittle.
    # maybe tool (ie end effector) should be part of the task config?

    if(client is None):
        client = bc.BulletClient(connection_mode=p.GUI if step_through else p.DIRECT)

    robots, obstacles = setup_environment(client=client)
    print("sending through bc client")
    print(client)
    ik_solutions = solve_ik(start_qs, target_poses, robots, obstacles, tool_name=TOOL_NAME, ee_collisions=True, step_through=False, client=client)

    # may need to force this to float?

    if len(ik_solutions) > 0:
        return ik_solutions
    else:
        print("Unable to solve requested target")