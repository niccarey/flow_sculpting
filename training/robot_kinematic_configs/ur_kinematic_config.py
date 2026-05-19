
# kinematic functions and constants specific to UR10e robot
# Relies on https://github.com/cambel/ur_ikfast

# See git repo for dependencies. Install with 
# git clone https://github.com/cambel/ur_ikfast.git
# cd ur_ikfast
# pip install -e .

import logging
from typing import List, Tuple, Dict #might not need
import os

import time
import numpy as np
import random
import math

import pb_utils as pbu
import pybullet_utils.bullet_client as bc
import pybullet as p

from ur_ikfast import ur_kinematics # must be installed


ur10e_arm = ur_kinematics.URKinematics('ur10e')

# Create collision checking filter for UR linkages
IGNORE_COLLISIONS = {(6, 9), (2, 10), (3, 10), (4, 10), (6, 10), (3, 9), (4, 9), (2, 11), (3, 11), (5, 11), (6, 11), (10, 11), (2, 12), (9, 10)}

# IK Solver using pybullet with UR context
# solves ONLY from flange - structure starting configuration and target poses accordingly.
# see ik_solve_wrapper in task configurations for adding gripper offsets, etc.

def solve_ik(start_qs, target_poses, robots, obstacles, tool_name, ee_collisions=True, step_through=False, client=None):

    random.seed(0)
    np.random.seed(0)

    new_client = False
    # we should now always receive a client from the config caller, but just in case ...
    if(client is None):
        print("re-initializing client")
        client = bc.BulletClient(connection_mode=p.GUI if step_through else p.DIRECT)
        new_client = True
    
    for robot, start_q in zip(robots, start_qs):
        pbu.set_joint_positions(robot, pbu.get_movable_joints(robot, client=client), start_q, client=client)

        
    ik_solutions = []
    randomize_seed = [False, False]
    max_attempts = 15000

    for i in range(max_attempts):
        
        if(len (ik_solutions) == len(robots)):
            p.disconnect()
            return ik_solutions

        robot_idx = len(ik_solutions)

        if(i % int(math.sqrt(max_attempts)) == 0):
            ik_solutions = []
            randomize_seed = [True, False]
            continue

        robot = robots[robot_idx]
        target_pose = target_poses[robot_idx]

        if(target_pose is None):
            ik_solutions.append(start_qs[robot_idx])
            continue

        link = pbu.link_from_name(robot, tool_name, client=client)
        joints = pbu.get_movable_joints(robot, client=client)
        ranges = [pbu.get_joint_limits(robot, joint, client=client) for joint in joints]

        # Start with the current joint positions and then randomize within limits after
        if(not randomize_seed[robot_idx]):
            initialization_sample = start_qs[robot_idx]
            randomize_seed[robot_idx] = True
        else:
            initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]
            
        pbu.set_joint_positions(robot, joints, initialization_sample, client=client)

        conf = p.calculateInverseKinematics(
            int(robot), link, target_pose[0], target_pose[1], 
            residualThreshold=0.00001, maxNumIterations=5000
        )
        
        lower, upper = list(zip(*ranges))
        if(not pbu.all_between(lower, conf, upper)):
            print("IK solution outside limits")
            continue
        
        assert len(joints) == len(conf)
        pbu.set_joint_positions(robot, joints, conf, client=client)

        contact_points = []
        for obstacle in obstacles:
            contact_points += p.getClosestPoints(bodyA=obstacle, bodyB=robot, distance = pbu.MAX_DISTANCE)

        for r_idx in range(len(ik_solutions)):
            contact_points += p.getClosestPoints(bodyA=robots[r_idx], bodyB=robot, distance = pbu.MAX_DISTANCE)
        

        all_joints = pbu.get_joints(robot, client = client)
        check_link_pairs = (
            pbu.get_self_link_pairs(robot, all_joints, IGNORE_COLLISIONS, client = client)
        )

        self_collision = False
        for link1, link2 in check_link_pairs:
            if pbu.pairwise_link_collision(robot, link1, robot, link2, client = client):
                print(link1, link2)
                self_collision = True

        if(self_collision):
            print("Self collision")
            continue

        # Print contact points if there are any
        if contact_points:
            print("Collision!")
            # time.sleep(0.5)
            continue

        pose = pbu.get_link_pose(robot, link, client=client)
        trans_diff, rot_diff = pbu.get_pose_distance(target_pose, pose)

        if(trans_diff < 0.001 and rot_diff < 0.01):
            ik_solutions.append(list(conf))
            continue
        else:
            print("IK Error: {}, {}".format(trans_diff, rot_diff))

    if(new_client): 
        client.disconnect()
    
    return None
