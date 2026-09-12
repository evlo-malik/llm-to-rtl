"""Portable numerical references for simulation. Never consumed by hardware."""
import json
from pathlib import Path
import numpy as np


def save_bundle(directory, value):
    directory=Path(directory)
    arrays={}
    def encode(x):
        if isinstance(x,np.ndarray):
            key=f'a{len(arrays)}'; arrays[key]=x
            return {'array':key}
        if isinstance(x,dict): return {k:encode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [encode(v) for v in x]
        if isinstance(x,np.generic): return x.item()
        return x
    tree=encode(value)
    (directory/'reference.json').write_text(json.dumps(tree,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(directory/'reference.npz',**arrays)


def load_bundle(directory):
    directory=Path(directory)
    with np.load(directory/'reference.npz',allow_pickle=False) as arrays:
        def decode(x):
            if isinstance(x,dict):
                if set(x)=={'array'}: return arrays[x['array']].copy()
                return {k:decode(v) for k,v in x.items()}
            if isinstance(x,list): return [decode(v) for v in x]
            return x
        return decode(json.loads((directory/'reference.json').read_text()))
