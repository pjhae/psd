import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ORI_IND: int = 2
    MANUAL_COLLISION: bool = True
    RADIUS: float = 0.4
    OBJBALL_TYPE: str = "hinge"
    VELOCITY_LIMITS: float = 6

    def __init__(
        self,
        xml_file="/home/jonghae/MAQ/envs/mujoco/assets/point_maze.xml",
        terminate_when_unhealthy=True,
        reset_noise_scale=0.2,
    ):
        utils.EzPickle.__init__(**locals())

        # For observation space
        high = np.inf * np.ones(8, dtype=np.float32)
        high[5:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = gym.spaces.Box(low, high)

        # For initialization variables
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._goal_pos = np.array([10,10])
        self.current_goal_index = 0
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)


    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()

        ## Do Simulation
        qpos = self.sim.data.qpos.copy()
        qpos[2] += action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction
        qpos[0] += np.cos(ori) * action[0]
        qpos[1] += np.sin(ori) * action[0]
        qvel = np.clip(self.sim.data.qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()

        xy_position_after = self.get_body_com("torso")[:2].copy() - np.array([-15, 20])

        goals = [np.array([35, -25]), np.array([0, -15])]

        # 첫 번째 목표에 대한 보상 계산
        if self.current_goal_index == 0:
            reward = -np.linalg.norm(goals[0] - xy_position_after)
            # 첫 번째 목표에 충분히 가까워졌는지 확인
            if np.linalg.norm(goals[0] - xy_position_after) < 7.5:  # 7.5 : 도달 거리 임계값
                self.current_goal_index = 1  # 첫 번째 목표에 도달했을 때 큰 보상 제공
                # print("Subgoal reached")

        # 두 번째 목표에 대한 보상 계산
        if self.current_goal_index == 1:
            reward = 50 - np.linalg.norm(goals[1] - xy_position_after)
            # 두 번째 목표에 충분히 가까워졌는지 확인
            if np.linalg.norm(goals[1] - xy_position_after) < 5:  # 도달 거리 임계값
                reward += 1000  # 두 번째 목표에 도달했을 때 큰 보상 제공
                # print("Final goal reached")

        done = False  

        observation = self._get_obs()
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),

        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        ############################For image observation###########################
        # # 1. For Training
        # camera_data = np.array(self.render("rgb_array", 84, 84, 0))
        # CHW = np.transpose(camera_data, (2, 0, 1))

        # print(camera_data, CHW)

        # # If you wanna check the input image
        # plt.imshow(camera_data)
        # plt.show()

        # ## 2. For rendering check
        # data = self._get_viewer("rgb_array").read_pixels(52, 52, depth=False)
        # CHW = np.transpose(data[::-1, :, :], (2, 0, 1))

        # obs_dct = {}
        # obs_dct['image'] = np.array(CHW) / 255.0
        # obs_dct['vector'] = np.concatenate([self.action_buffer, [self.state_vector()[4]]])
        # #########################################################################
        # print(self.sim.data.qpos[:2])
        
        observations = np.concatenate([self._goal_pos, position, velocity])

        return observations
    
    def set_state_rgb(self, sampled_state):
        qpos = np.array(sampled_state[:3])  # 위치 정보
        qvel = np.array(sampled_state[3:])  # 속도 정보
        self.set_state(qpos, qvel)

        return self.render(mode='rgb_array', camera_id=0)
    

    def reset_model(self):
        self.current_goal_index = 0

        qpos = self.init_qpos 
        
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        # option2
        # self._goal_pos = np.random.uniform(low=-8.0, high=8.0, size=2)

        # self.sim.model.geom_pos[box_id][0:2] = self._goal_pos 
        # self.sim.model.geom_pos[box_id][0:2] = np.array([100,100])
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)