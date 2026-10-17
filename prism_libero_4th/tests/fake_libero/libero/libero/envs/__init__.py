"""TEST ONLY deterministic physics-free dummy environment."""
import numpy as np

class OffScreenRenderEnv:
    def __init__(self, bddl_file_name, camera_heights=128, camera_widths=128):
        self.env = self
        self.action_spec = (-np.ones(7), np.ones(7))
        self.n = 0
        self.camera_heights=camera_heights
        self.camera_widths=camera_widths
    def seed(self, seed): pass
    def reset(self):
        self.n=0
        return self._obs()
    def set_init_state(self, state):
        self.n=0
        return self._obs()
    def _obs(self):
        im = np.zeros((self.camera_heights,self.camera_widths,3),dtype=np.uint8)
        im[..., 0] = min(255, 25 + self.n*8)
        return {'agentview_image': im, 'robot0_eye_in_hand_image': im,
                'robot0_eef_pos': np.array([self.n*.01,0,.2]),
                'thing_1_pos': np.array([.1,0,.03]),
                'robot0_gripper_qpos': np.array([.04,-.04])}
    def step(self, action):
        assert len(action)==7 and np.isfinite(action).all()
        assert np.max(np.abs(action)) <= 1+1e-6
        self.n += 1
        return self._obs(), float(self.check_success()), self.n >= 12, {}
    def check_success(self): return self.n >= 11
    def close(self): pass
