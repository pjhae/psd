
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
    
    register(
        id="Hopper-v2",
        entry_point="envs.mujoco:HopperEnv",
        max_episode_steps=1000,
        reward_threshold=3800.0,
    )

    register(
	    id="Hopper-v3",
	    entry_point="envs.mujoco.hopper_v3:HopperEnv",
	    max_episode_steps=1000,
	    reward_threshold=3800.0,
	)
	
    register(
        id="Walker2d-v2",
        max_episode_steps=1000,
        entry_point="envs.mujoco:Walker2dEnv",
    )

    register(
        id="Walker2d-v3",
        max_episode_steps=1000,
        entry_point="envs.mujoco.walker2d_v3:Walker2dEnv",
    )

    register(
        id="Humanoid-v2",
        entry_point="envs.mujoco:HumanoidEnv",
        max_episode_steps=1000,
    )

    register(
        id="Humanoid-v3",
        entry_point="envs.mujoco.humanoid_v3:HumanoidEnv",
        max_episode_steps=1000,
    )
