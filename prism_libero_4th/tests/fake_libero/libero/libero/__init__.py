"""TEST ONLY LIBERO API mock. Never use this package for a real benchmark."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np


class Suite:
    n_tasks = 1
    def get_task(self, task_id):
        return SimpleNamespace(name='task0_pick_up_thing', language='dummy task',
                               problem_folder='fake', bddl_file='task0.bddl')
    def get_task_init_states(self, task_id):
        return [np.zeros(3, np.float32)]


class benchmark:
    @staticmethod
    def get_benchmark_dict():
        return {'libero_spatial': Suite}


def get_libero_path(kind):
    return str(Path(__file__).resolve().parent/'bddl_files')
