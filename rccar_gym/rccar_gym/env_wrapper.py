
import numpy as np
import gymnasium as gym
import rccar_gym # rccar_gym을 import 해야 gym.make에서 환경을 찾을 수 있습니다.

class RCCarWrapper(gym.Wrapper):
    def __init__(self, args=None, maps=None, render_mode=None) -> None:
        self._env = gym.make("rccar-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        
        # 라이다 각도 범위를 스캔 인덱스로 변환
        start, end = (np.array(args.lidar_range) * 4 + 540).astype(int)
        self.start, self.end = max(0, start), min(1080, end)

        # 환경의 핵심 파라미터 저장
        self.num_agents = self._env.unwrapped.num_agents
        self.num_controlled_agents = self._env.unwrapped.num_controlled_agents

        self.track = self._env.unwrapped.track
        self.waypoints = np.stack([self.track.centerline.xs, self.track.centerline.ys]).T
        
        # 체크포인트 관련 설정 (reset 시 초기화)
        self.checkpoints = [[] for _ in range(self.num_controlled_agents)]
        self.n_checkpoints = 20
        self.next_checkpoints = np.zeros(self.num_controlled_agents, dtype=int)
        self.arrive_distance = 1.5
        self.checkpoint_distances = np.zeros(self.num_controlled_agents, dtype=np.float32)

        self.collisions = np.zeros(self.num_controlled_agents)
        self.reward = np.zeros(self.num_controlled_agents)
        self.prev_checkpoints = np.zeros(self.num_controlled_agents)
        self.prev_checkpoint_dists = np.zeros(self.num_controlled_agents)
        # 차량 제어 한계값
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

    def _process_obs(self, obs_dict, info):
        """
        리워드에 필요한 값 계산 및 변수 업데이트에 사용
        """
        processed_obs_list = []
        for i in range(self.num_controlled_agents):
            pos = np.array([obs_dict['poses_x'][i], obs_dict['poses_y'][i]])
            yaw = obs_dict['poses_theta'][i]
            scan = obs_dict['scans'][i][self.start:self.end]
            processed_obs_list.append([pos, yaw, scan])

        current_poses = np.stack([obs_dict['poses_x'], obs_dict['poses_y']]).T
        for i in range(self.num_controlled_agents):
            if self.next_checkpoints[i] != self.n_checkpoints:
                checkpoint_pos = self.checkpoints[i][self.next_checkpoints[i]]
                checkpoint_dist = np.linalg.norm(current_poses[i] - checkpoint_pos)
                if checkpoint_dist < self.arrive_distance:
                    self.next_checkpoints[i] += 1
                    checkpoint_pos = self.checkpoints[i][self.next_checkpoints[i]]
                    checkpoint_dist = np.linalg.norm(current_poses[i] - checkpoint_pos)
                self.checkpoint_distances[i] = checkpoint_dist

        self.collisions = obs_dict["collisions"]

        info['waypoints_passed'] = self.next_checkpoints

        return processed_obs_list, info

    def reset(self, **kwargs):
        obs_dict, info = self._env.reset(**kwargs)

        self.next_checkpoints = np.zeros(self.num_controlled_agents, dtype=int)
        self.checkpoints = [[] for _ in range(self.num_controlled_agents)]

        self.checkpoint_distances = np.zeros(self.num_controlled_agents, dtype=np.float32)

        self.reward = np.zeros(self.num_controlled_agents)
        initial_poses = np.stack([obs_dict['poses_x'], obs_dict['poses_y']]).T

        for i in range(self.num_controlled_agents):
            agent_pos = initial_poses[i]
            dists = np.linalg.norm(self.waypoints - agent_pos, axis=-1)
            init_idx = np.argmin(dists)
            N = self.waypoints.shape[0]
            for j in range(self.n_checkpoints):
                chk_idx = (int(N * (j + 1) / self.n_checkpoints) - 1 + init_idx) % N
                self.checkpoints[i].append(self.waypoints[chk_idx])

        processed_obs, info = self._process_obs(obs_dict, info)

        return processed_obs, info

    def step(self, actions: np.ndarray):
        """actions: shape (num_agents, 2)의 numpy 배열"""
        
        clipped_actions = np.zeros((self.num_agents, 2))
        clipped_actions[:self.num_controlled_agents, 0] = np.clip(actions[:self.num_controlled_agents, 0], -self.max_steer, self.max_steer)
        clipped_actions[:self.num_controlled_agents, 1] = np.clip(actions[:self.num_controlled_agents, 1], self.min_speed, self.max_speed)

        obs_dict, _, terminate, truncate, info = self._env.step(clipped_actions)

        # 관측값 처리
        processed_obs, info = self._process_obs(obs_dict, info)

        # 리워드 처리
        self.compute_reward()

        return processed_obs, self.reward, terminate, truncate, info
    
    def compute_reward(self):
        self.reward = None # update reward

        
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

