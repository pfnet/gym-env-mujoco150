import numpy as np
from gym_env_mujoco150 import mujoco_env
from gym import utils
import mujoco_py

def mass_center(model,data):
    mass = model.body_mass
    xpos = data.xipos
    poss = [ xpos[i] * mass[i] for i in range(len(mass))]
    return (np.sum(poss, 0) / np.sum(mass))[0]

class OP3FullbodyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'robotis_op3_fullbody.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a):
        pos_before = mass_center(self.model,self.sim.data)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model,self.sim.data)
        
        alive_bonus = 5.0
        data = self.sim.data
        # 3.2 is humanoid(1.6m) / op3(500mm) 
        lin_vel_cost = 0.25 * 3.2 * (pos_after - pos_before) / self.model.opt.timestep
        # 0.4 is humanoid(maxctrl=0.4) / op3(maxctrl=1.0)
        quad_ctrl_cost = 0.1 * np.square(data.ctrl * 0.4).sum()
        # 17.14 is humanoid(60kg) / op3(3.2kg)
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext * 17.14).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos

        done = bool((qpos[2] < 0.15) or (qpos[2] > 0.5))
        if done :
            print('done :', qpos[2])
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = mujoco_py.const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.lookat[2] = .5
        self.viewer.cam.elevation = -10

