import os
import time
import argparse
import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import cv2

# 수정된 RCCarWrapper import
from rccar_gym.env_wrapper import RCCarWrapper

TEAM_NAME = "HS_Multi" # 팀 이름 변경

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--num_controlled_agents', default=2, type=int, help="Number of agents to simulate.")
    parser.add_argument('--num_static_agents', default=8, type=int, help="Number of static agents as obstacle.")
    parser.add_argument('--num_dynamic_agents', default=0, type=int, help="Number of dynamic agents as obstacle.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str)
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str) 
    parser.add_argument("--render", default=True, action='store_true')
    parser.add_argument("--no_render", action='store_false', dest='render')

    args = parser.parse_args()
    args = EasyDict(vars(args))

    ws_path = "/workspace/IIS-TEST"
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)

    with open(args.env_config, 'r') as f:
        env_args = EasyDict(YAML().load(f))
    
    with open(args.dynamic_config, 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(env_args)
    args.update(dynamic_args)
    
    args.num_agents = args.num_controlled_agents + args.num_static_agents + args.num_dynamic_agents
    if args.num_controlled_agents == 1:
        args.collision_reset = False
    else:
        args.collision_reset = True
    
    # 맵 목록 찾기
    args.maps_dir = os.path.join(ws_path, 'maps')
    args.maps = [map_name for map_name in os.listdir(args.maps_dir) if os.path.isdir(os.path.join(args.maps_dir, map_name))]
    
    return args

class SimplePIDController:
    """각 에이전트를 위한 간단한 PID/Pure Pursuit 컨트롤러"""
    def __init__(self, waypoints, args, Kp=0.8, Ki=0.001, Kd=0.1):
        self.waypoints = waypoints
        self.N = waypoints.shape[0]
        self.lookahead = 10
        
        # PID 계수
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # PID 상태 변수
        self.prev_err = 0.0
        self.sum_err = 0.0
        
        # 환경 파라미터
        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

    def get_action(self, agent_obs):
        pos, yaw, _ = agent_obs

        dists = np.linalg.norm(self.waypoints - pos, axis=1)
        nearest_idx = np.argmin(dists)

        lookahead_idx = (nearest_idx + self.lookahead) % self.N
        lookahead_point = self.waypoints[lookahead_idx]

        relative_point = lookahead_point - pos
        rotated_point_x = relative_point[0] * np.cos(-yaw) - relative_point[1] * np.sin(-yaw)
        rotated_point_y = relative_point[0] * np.sin(-yaw) + relative_point[1] * np.cos(-yaw)
        
        curr_err = np.arctan2(rotated_point_y, rotated_point_x)
        
        self.sum_err += curr_err * self.dt
        derivative_err = (curr_err - self.prev_err) / self.dt
        
        steer = self.Kp * curr_err + self.Ki * self.sum_err + self.Kd * derivative_err
        self.prev_err = curr_err

        speed = self.max_speed * (1.0 - 0.8 * abs(steer) / self.max_steer)
        
        return np.array([steer, speed])


def main():
    args = get_args()
    
    target_map = 'map1'
    if target_map not in args.maps:
        print(f"Error: Map '{target_map}' not found in maps directory.")
        return

    render_mode = "human_fast" if args.render else None
    
    env = RCCarWrapper(args=args, maps=[target_map], render_mode=render_mode)
    
    waypoints = env.waypoints
    controllers = [SimplePIDController(waypoints, args, np.random.uniform(low=0.1, high=1.0), np.random.uniform(low=0.001, high=0.01), np.random.uniform(low=0.1, high=1.0)) for _ in range(env.num_controlled_agents)]
    # controllers = [SimplePIDController(waypoints, args) for _ in range(env.num_controlled_agents)]

    obs_list, info = env.reset()

    start_time = time.time()
    step = 0
    max_steps = 10000 

    while step < max_steps:
        actions = []
        for i in range(env.num_controlled_agents):
            agent_obs = obs_list[i]
            action = controllers[i].get_action(agent_obs)
            
            actions.append(action)
        
        actions_np = np.array(actions)
        
        obs_list, reward, terminate, truncate, info = env.step(actions_np)
        
        if args.render:
            env.render()
        
        if terminate or truncate:
            print(f"Simulation finished at step {step}.")
            if info.get('collisions', np.any(env._env.unwrapped.collisions)):
                 print("Reason: Collision")
            elif truncate:
                 print("Reason: Truncated (max steps reached)")
            else:
                 print("Reason: Lap finished")
            break
        
        step += 1

    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds.")
    env.close()


if __name__ == '__main__':
    main()