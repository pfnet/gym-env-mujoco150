# gym-env-mujoco150
OpenAI-gym mujoco environment with mujoco1.50 and mujoco-py1.50.

## Example
```
import gym_env_mujoco150

env = gym_env_mujoco150.make('InvertedPendulum-mujoco150-v1')
env.reset()
env.render()
```

## Requirement
mujoco version 1.50

* gym 0.9.2
* mujoco-py 1.50.1

# License

MIT License (see `LICENSE.md` file).
