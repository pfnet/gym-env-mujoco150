from gym_env_mujoco150.mujoco_env import MujocoEnv
from gym_env_mujoco150.inverted_pendulum import InvertedPendulumEnv
from gym_env_mujoco150.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym_env_mujoco150.hopper import HopperEnv
from gym_env_mujoco150.swimmer import SwimmerEnv
from gym_env_mujoco150.walker2d import Walker2dEnv
from gym_env_mujoco150.humanoid import HumanoidEnv
from gym_env_mujoco150.op3_fullbody import OP3FullbodyEnv

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
    id='Swimmer-mujoco150-v1',
    entry_point='gym_env_mujoco150:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-mujoco150-v1',
    max_episode_steps=1000,
    entry_point='gym_env_mujoco150:Walker2dEnv',
)

register(
    id='Humanoid-mujoco150-v1',
    entry_point='gym_env_mujoco150:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='OP3Fullbody-v1',
    entry_point='gym_env_mujoco150:OP3FullbodyEnv',
    max_episode_steps=1000,
)
