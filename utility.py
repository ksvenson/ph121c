import os
import numpy as np
import pickle
from typing import Literal

cache_type = Literal['npy', 'npz', 'pkl']

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}


class PhysicsError(Exception):
    pass


def cache(method: cache_type, base):
    def wrap(func):
        def inner(*args, **kwargs):
            fname = base
            if 'note' in kwargs:
                if kwargs['note'] is not None:
                    fname += '_' + kwargs['note']
            fname += f'.{method}'
            if os.path.isfile(fname):
                if method == 'npy':
                    data = np.load(fname)
                elif method == 'npz':
                    data = dict(np.load(fname))
                elif method == 'pkl':
                    with open(fname, 'rb') as file:
                        data = pickle.load(file)
            else:
                data = func(*args, **kwargs)
                if method == 'npy':
                    np.save(fname, data)
                elif method == 'npz':
                    np.savez(fname, **data)
                elif method == 'pkl':
                    with open(fname, 'wb') as file:
                        pickle.dump(data, file)
            return data
        return inner
    return wrap
