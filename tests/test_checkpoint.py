import json
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from compiler.checkpoint import load_checkpoint


def test_sharded_checkpoint_and_provenance(tmp_path):
    (tmp_path/'config.json').write_text('{"model_type":"gpt2"}')
    save_file({'a':torch.tensor([1.,2.])},str(tmp_path/'one.safetensors'))
    save_file({'b':torch.tensor([3.])},str(tmp_path/'two.safetensors'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'one.safetensors','b':'two.safetensors'}}))
    cfg,state,hashes=load_checkpoint(tmp_path)
    assert cfg['model_type']=='gpt2' and set(state)=={'a','b'}
    assert set(hashes)=={'config.json','one.safetensors','two.safetensors'}
    assert all(len(digest)==64 for digest in hashes.values())


def test_reject_path_escape(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'../outside.safetensors'}}))
    with pytest.raises(ValueError,match='outside'):
        load_checkpoint(tmp_path)


def test_reject_nonfinite(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    save_file({'a':torch.tensor([float('nan')])},str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='non-finite'):
        load_checkpoint(tmp_path)
