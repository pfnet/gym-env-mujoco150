from gym_env_mujoco150.mujoco_env import MujocoEnv
from gym_env_mujoco150.inverted_pendulum import InvertedPendulumEnv
from gym_env_mujoco150.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym_env_mujoco150.hopper import HopperEnv
from gym_env_mujoco150.humanoid import HumanoidEnv

from gym.envs.registration import registry, register, make, spec

register(
    id='InvertedPendulum-mujoco150-v1',
    entry_point='gym_env_mujoco150:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-mujoco150-v1',
    entry_point='gym_env_mujoco150:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='Hopper-mujoco150-v1',
    entry_point='gym_env_mujoco150:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Humanoid-mujoco150-v1',
    entry_point='gym_env_mujoco150:HumanoidEnv',
    max_episode_steps=1000,
)
