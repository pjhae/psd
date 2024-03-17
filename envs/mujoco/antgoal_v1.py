import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
        self,
        xml_file="/home/jonghae/MAQ/envs/mujoco/assets/ant.xml", 

        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
    ):
        utils.EzPickle.__init__(**locals())

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.goal_pos = np.array([0, 0])

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def step(self, action):

        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
	
        reward = -np.linalg.norm(self.goal_pos - xy_position_after)

        if self.get_body_com("torso")[2].copy() > 0.2 and self.get_body_com("torso")[2].copy() < 1.0:
            reward += 1
        
        if np.linalg.norm(self.goal_pos - xy_position_after) < 2:
            # print("goal in")
            reward += 7.5

        done = False

        observation = self._get_obs()
        info = {

            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,

        }

        return observation, reward, done, info

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((self.goal_pos, position, velocity))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        # Randomize goal
        index = np.random.randint(0, 6)
        angle = np.deg2rad(index * 60)
        self.goal_pos = np.array([10*np.cos(angle), 10*np.sin(angle)])
        # print(index)

        self.set_state(qpos, qvel)
        observation = self._get_obs()

        return observation

    def set_state_rgb(self, sampled_state):
        qpos = np.array(sampled_state[:15])  # 위치 정보
        qvel = np.array(sampled_state[15:])  # 속도 정보
        self.set_state(qpos, qvel)

        return self.render(mode='rgb_array', camera_id=0)

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
                
                
