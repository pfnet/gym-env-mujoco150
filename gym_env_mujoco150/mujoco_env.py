import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
    from mujoco_py import MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, model_path, frame_skip):
        print(model_path)
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.timestep = self.model.opt.timestep * self.frame_skip
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        # changed a place of setting action space
        # sometimes use self.action_space at first call self._step()
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            # TODO:
            # self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos[:]
        sim_state.qvel[:] = qvel[:]
        self.sim.set_state(sim_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        assert len(ctrl) is len(self.sim.data.ctrl)
        self.data.ctrl[:] = ctrl[:]
        for _ in range(n_frames):
            self.sim.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer = None
            return

        if mode == 'rgb_array':
            pass
            # TODO:
            # self._get_viewer().render()
            # data, width, height = self._get_viewer()._read_pixels_as_in_window()#get_image()
            # return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
            # return self._get_viewer()._read_pixels_as_in_window()#get_image()
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xipos(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_cvel(self.model.body_name2id(body_name))

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name).reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])
