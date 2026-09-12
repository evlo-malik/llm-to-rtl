"""Local safetensors checkpoints, including Hugging Face shard indexes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def load_checkpoint(directory):
    directory=Path(directory).resolve()
    cfg=json.loads((directory/'config.json').read_text())
    index=directory/'model.safetensors.index.json'
    if index.exists():
        weight_map=json.loads(index.read_text())['weight_map']
        files=sorted(set(weight_map.values()))
    else:
        weight_map=None
        files=['model.safetensors']
    state={}; hashes={}
    for name in ['config.json']+files:
        path=(directory/name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('checkpoint index references a file outside the model directory')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        hashes[name]=digest.hexdigest()
        if name=='config.json': continue
        for key,value in load_file(str(path)).items():
            if key in state: raise ValueError(f'duplicate tensor: {key}')
            state[key]=value.float().numpy()
            if not np.isfinite(state[key]).all(): raise ValueError(f'non-finite tensor: {key}')
            if weight_map is not None and weight_map.get(key)!=name:
                raise ValueError(f'shard index mismatch: {key}')
    if weight_map is not None and set(weight_map)!=set(state):
        raise ValueError('checkpoint index and tensors disagree')
    return cfg,state,hashes
