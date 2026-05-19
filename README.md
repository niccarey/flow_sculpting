# flow_sculpting

Streamlined implementation of go-flow training with SVF-based reward functions. This code repo is a functional skeleton only - URDF models, trained checkpoints, etc, are not included, and the relative folder heirarchy used for file referencing may not be up to date.

See curtisa GO-FLOW repository for an (older) version with with RL-games configuration and requirements / environment details.

This repository contains a refactored go-flow implementation which attempts to simplify the process of running an already-trained policy using a proprietary digital twin library to control the hardware (in simulation or in real). This component will not function as a stand-alone.


**Other info**

Environments can be fairly generalized, we have set up one environment to handle single-arm configurations and another for multi-arm tasks. However, policy runners are task-specific, they reference task config files and specify the parameter domains based on the geonetry and bounds of the insertion.

When setting up a new environment and task configuration, ensure policy runner references the correct environment, environment id, and config file. Ditto gym environment class

Demo scripts are minimal scripts for running a pre-trained policy. Admittance controller is a legacy control wrapper for UR robots, replace calls to this with the hardware control wrapper of your choice.

Training contains training-relevant files, Execution contains execution-relevant files.
pb_utils is legacy and can probably be removed / absorbed into other files or classes.


File /content descriptions:

- rl_components: as per multi-robot-assembly (no change)
- bolt_gear_insertion_config.yaml: Configuration file for the shared context components and runner network.
- demo_twoarm_assembly.py : a minimal script using araas to move to a pre-insertion point and trigger a pretrained insertion policy, while using araas to send commands derived from policy actions to the hardware
- two_arm_insertion_environment.py : example of a gym.Env-derived minimal class that acts as the interface (stepping through / resetting) for the RL player network. Registering this environment with the RL_Games environment overwrites the step functions in the RL_Games 'player' class - the same player network could instead step through an IsaacGym environment (for example) by using a different gym.Env.
- demo_elbow_assembly.py : a draft of the elbow-to-peg insertion task (araas side)
- simple_insertion_environment.py : minimal class for one robot doing a cartesian-space-only insertion

Notes on demo_twoarm_assembly: 

Testing script only - hardcodes a lot of workcell environment variables (goal positions, grasp positions, etc) that previously were wrapped in a TaskConfig translation layer. Eventually these should be replaced by perception layer inputs. 

Demo structure:
- initialise workcell
- orchestrate the pick up and pre-configuration motions for a peg-in-hole two-handed insertion (hole == gear / peg == bolt) 
- set up the araas side of a REP/REQ messaging socket, to handle Action messages coming from the runner
- run robot control script (waits for Actions, calculates corresponding controller command, distributes to robots, gets updated robot state, sends to runner)

Notes on insertion environment:

Bare minimum requirements for gym framework:
```
gym.Env.step(self, action: ActType)
gym.Env.reset(self, *, seed: int | None = None, options: dict | None = None) → Tuple[ObsType, dict]
gym.Env.render(self) → RenderFrame | List[RenderFrame] | None #(this should be none for running on araas)
Env.action_space: Space[ActType]
Env.observation_space: Space[ObsType]
Env.reward_range = (-inf, inf)
gym.Env.close(self) # can be omitted 
```

Function set_up_spaces is called any time before we invoke step (can also be called from outside)

