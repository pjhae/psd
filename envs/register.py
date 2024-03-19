
from gym.envs.registration import register

def register_custom_envs():

    register(
        id="Ant-v3",
        entry_point='envs.mujoco.ant_v3:AntEnv',
        max_episode_steps=200,
        reward_threshold=6000,
    )


    register(
        id="Point-v1",
        entry_point='envs.mujoco.point_v1:PointEnv',
        max_episode_steps=200,
        reward_threshold=6000,
    )