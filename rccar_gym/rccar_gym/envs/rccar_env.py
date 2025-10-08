# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Hongrui Zheng
Fix: Hyeokjin Kwon - https://github.com/ineogi2
"""

# gym imports
import gymnasium as gym
from typing import List

from .action import (CarAction, from_single_to_multi_action_space)
from .integrator import IntegratorType
from .rendering import make_renderer

from .track import Track

# base classes
from .base_classes import Simulator, DynamicModel
from .observation import observation_factory
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update


# others
import numpy as np
import random


class RCCarEnv(gym.Env):
    """
    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment.

            params (dict)
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=1): number of agents in the environment

            timestep (float, default=0.025): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """

    # NOTE: change matadata with default rendering-modes, add definition of render_fps
    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 100}

    def __init__(self, args, maps, render_mode=None):
        super().__init__()
        self.seed = args.seed

        # Configuration
        self.config = self.set_config(args)
        self.maps = maps
        self.params = self.config["params"]
        
        self.num_controlled_agents = self.config["num_controlled_agents"]
        self.num_static_agents = self.config["num_static_agents"]
        self.num_dynamic_agents = self.config["num_dynamic_agents"]
        self.num_agents = self.config["num_agents"]

        self.timestep = self.config["timestep"]
        self.ego_idx = self.config["ego_idx"]
        self.integrator = IntegratorType.from_string(self.config["integrator"])
        self.model = DynamicModel.from_string(self.config["model"])
        self.observation_config = self.config["observation_config"]
        self.action_type = CarAction(self.config["control_input"], params=self.params)
        self.max_episode_steps = 10000 #self.config["max_episode_steps"]
        self.collision_reset = self.config["collision_reset"]

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        self.active_agents = []

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.step_count = 0
        self.lap_times = np.zeros((self.num_controlled_agents,))
        self.lap_counts = np.zeros((self.num_controlled_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_controlled_agents)
        self.toggle_list = np.zeros((self.num_controlled_agents,))
        self.start_xs = np.zeros((self.num_controlled_agents,))
        self.start_ys = np.zeros((self.num_controlled_agents,))
        self.start_thetas = np.zeros((self.num_controlled_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.num_static_agents,
            self.num_dynamic_agents,
            self.collision_reset,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            model=self.model,
            action_type=self.action_type,
        )
        self.sim.update_params(self.config["params"])

        self.init_map()

        # render config
        self.render_obs = None
        self.render_mode = render_mode
        self.render_color = np.concatenate((np.zeros(self.num_controlled_agents), np.ones(self.num_agents-self.num_controlled_agents)),axis=0)
        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x

        # action space
        self.action_space = from_single_to_multi_action_space(
            self.action_type.space, self.num_agents
        )
        
    def set_config(self, args) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().

        Args:
            None

        Returns:
            a configuration dict
        """
        return {
            "seed": args.seed,
            "params": {
                "mu": args.mu,
                "C_Sf": args.C_Sf,
                "C_Sr": args.C_Sr,
                "lf": args.length_f,
                "lr": args.length_r,
                "h": args.h,
                "m": args.m,
                "I": args.I,
                "s_min": args.s_min,
                "s_max": args.s_max,
                "sv_min": args.sv_min,
                "sv_max": args.sv_max,
                "v_switch": args.v_switch,
                "a_max": args.a_max,
                "v_min": args.v_min,
                "v_max": args.v_max,
                "width": args.width,
                "length": args.length,
                "lidar_pos_offset": args.lidar_pos_offset
            },
            "num_agents": args.num_agents,
            "num_controlled_agents": args.num_controlled_agents, 
            "num_static_agents": args.num_static_agents,
            "num_dynamic_agents": args.num_dynamic_agents,
            "timestep": args.timestep,
            "ego_idx": args.ego_idx,
            "integrator": args.integrator,
            "model": args.model,
            "control_input": list(args.control_input),
            "observation_config": args.observation_config,
            "collision_reset": args.collision_reset,
            "reset_config": args.reset_config,
            "max_episode_steps": args.max_episode_steps
        }

    def init_map(self, map=None):
        if map is None:
            idx = random.randint(0, len(self.maps)-1)
            map = self.maps[idx]
        assert map in self.maps
        self.map = map
        self.sim.set_map(self.map)

        # load track in gym env for convenience
        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(self.map)
        
        # waypoint
        self.waypoints = np.stack(
            [self.track.centerline.xs, self.track.centerline.ys, self.track.centerline.vxs]
        ).T

        ## random positioning of obstacles -> may need fixing for contest in order to fix the positions/path 
        self.dynamic_poses = None
        self.static_poses = None  # 초기화
        self.dynamic_paths = []

        if self.num_agents > 1:
            center_xs = self.track.centerline.xs
            center_ys = self.track.centerline.ys
            track_width = self.track.spec.width       

            dx = np.diff(center_xs, append=center_xs[0]) # 마지막 점은 첫 점을 향하도록 설정 (순환 트랙)
            dy = np.diff(center_ys, append=center_ys[0])
            thetas = np.arctan2(dy, dx)
            offset_dist = track_width / 2.0 * 0.7
            
            right_xs = center_xs - offset_dist * np.sin(thetas)
            right_ys = center_ys + offset_dist * np.cos(thetas)

            left_xs = center_xs + offset_dist * np.sin(thetas)
            left_ys = center_ys - offset_dist * np.cos(thetas)
            
            if self.num_static_agents > 0:
                all_poses = []

                #waypoint를 점 50개 간격으로 sampling 한 후 그중에서 num_static 만큼 선택
                grouped_index = np.array([i for i in range(0, len(center_xs), 50)])
                indices = np.sort(np.random.choice(a=len(grouped_index), size=self.num_static_agents, replace=False))
                indices = grouped_index[indices]

                #static의 위치를 left, right center중 랜덤하게 스폰
                np.random.shuffle(indices)

                #include center spawn
                # spawn_prob = [1/3, 1/3, 1/3]
                
                # exclude center spawn
                static_prob = [1/2, 1/2, 0]
                
                group_sizes = np.random.multinomial(len(indices), static_prob)
                left, right, center = np.split(indices, np.cumsum(group_sizes)[:-1])
                print("STATIC OBSTACLE SPAWN")
                print(f"left: {len(left)}, right: {len(right)}, center: {len(center)}")

                if len(left) > 0:
                    poses_left = np.stack((left_xs[left], left_ys[left], thetas[left]), axis=1)
                    all_poses.append(poses_left)

                if len(right) > 0:
                    poses_right = np.stack((right_xs[right], right_ys[right], thetas[right]), axis=1)
                    all_poses.append(poses_right)

                if len(center) > 0:
                    poses_center = np.stack((center_xs[center], center_ys[center], thetas[center]), axis=1)
                    all_poses.append(poses_center)
                
                self.static_poses = np.vstack(all_poses)

            if self.num_dynamic_agents > 0:
                dynamic_poses = []
                self.dynamic_paths = []
                for _ in range(self.num_dynamic_agents):
                    
                    # include center spawn of dynamic agents
                    # dir = np.random.choice(np.arange(1,4), 1, replace=False).item()
                    
                    # exclude center spawn of dynamic agents
                    dynamic_dir = np.random.choice(np.arange(1,3), 1, replace=False).item()

                    if dynamic_dir == 1:
                        path = np.column_stack((left_xs, left_ys, thetas))
                    elif dynamic_dir == 2:
                        path = np.column_stack((right_xs, right_ys, thetas))
                    else:
                        path = np.column_stack((center_xs, center_ys, thetas))

                    path = np.flip(path, axis=0)
                    start_index = np.random.choice(len(center_xs), 1, replace=False).item()
                    start_pose = path[start_index]
                    dynamic_poses.append([start_pose])
                    self.dynamic_paths.append([start_index, path])
                    
                self.dynamic_poses = np.vstack(dynamic_poses)

        self.sim.static_poses = self.static_poses
        self.sim.dynamic_paths = self.dynamic_paths

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        
        assert (
            "type" in self.observation_config
        ), "observation_config must contain 'type' key"
        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=1
        )
    
        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=self.render_mode,
            render_fps=self.metadata["render_fps"],
        )

    def configure(self, config: dict) -> None:
        if config:
            self.config = deep_update(self.config, config)
            self.params = self.config["params"]

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1

        for i in range(self.num_controlled_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        terminate = np.sum(self.collisions)>0 or np.all(self.toggle_list[:self.num_controlled_agents] >= 2)
        truncate = (self.current_time // self.timestep) >= self.max_episode_steps
        terminate, truncate = bool(terminate), bool(truncate)

        return terminate, truncate, self.toggle_list >= 2

    def _update_state(self):
        """
        Update the env's states according to observations.
        """
        self.poses_x = self.sim.agent_poses[:, 0]
        self.poses_y = self.sim.agent_poses[:, 1]
        self.poses_theta = self.sim.agent_poses[:, 2]
        self.collisions = self.sim.collisions

    def step(self, action:np.ndarray):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        # action = action.reshape(-1, 2)
        # action = np.array([action]) 
        self.sim.step(action)

        # observation
        obs = self.observation_type.observe()
        # times
        reward = 0.0     # not use
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "velocity" : self.sim.agent_velocity,
            "steering_angles": self.sim.agent_steerings,
            "action": action,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
            "render_color": self.render_color
        }

        # check done
        terminate, truncate, toggle_list = self._check_done()
        info = {"checkpoint_done": toggle_list[:self.num_controlled_agents]}

        return obs, reward, terminate, truncate, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        if options is not None:
            self.init_map(options['map'])
        else:
            self.init_map() 

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        poses = np.zeros((self.num_agents, 3))
        
        if options is not None and "poses" in options:
            init_pose = options["poses"]
            self.static_poses = options["static_poses"]
            self.sim.static_poses = self.static_poses
            self.dynamic_paths = options["dynamic_paths"]
            self.sim.dynamic_paths = self.dynamic_paths
        else:
            init_pose = self.reset_fn.sample()
            poses[:self.num_controlled_agents, :] = np.repeat(init_pose, repeats=self.num_controlled_agents, axis=0)
        
        poses[self.num_controlled_agents:self.num_controlled_agents+self.num_static_agents, :] = self.static_poses
        poses[self.num_controlled_agents+self.num_static_agents:,:] = self.dynamic_poses
        
        self.sim.controlled_init_pose = poses[0]

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        return obs, info

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        self.sim.set_map(map_name)
        self.track = Track.from_track_name(map_name)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()
