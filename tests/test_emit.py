import numpy as np
import pytest
from compiler.emit import csd, emit_linear


def test_csd_signed_int8_exhaustive():
    for value in range(-128, 128):
        assert sum(sign * 2**shift for sign, shift in csd(value)) == value


def test_reject_overflow_and_noninteger(tmp_path):
    with pytest.raises(ValueError, match="overflow"):
        emit_linear(tmp_path / "bad.sv", "bad", np.array([[127]]), [2**31-1])
    with pytest.raises(ValueError, match="integer"):
        emit_linear(tmp_path / "bad.sv", "bad", np.array([[1.1]]))


def test_no_weight_storage_or_runtime_multiplier(tmp_path):
    p = tmp_path / "fixed.sv"
    emit_linear(p, "fixed", np.array([[7, -7], [0, 1], [7, -7]]))
    text = p.read_text()
    assert "$readmem" not in text
    assert "load_w" not in text
    assert "*" not in text
    assert "ROM" not in text
    assert "p0_p7" in text


def test_synthesis_has_no_weight_memory_or_multiplier(tmp_path):
    import json
    import shutil
    import subprocess
    yosys=shutil.which('yosys')
    if not yosys:
        pytest.skip('yosys is not installed')
    p=tmp_path/'fixed.sv'
    net=tmp_path/'fixed.json'
    emit_linear(p,'fixed',np.array([[7,-7,0],[-128,127,1]]),[3,-9])
    script=f'read_verilog -sv {p}; hierarchy -top fixed; proc; flatten; opt; memory_map; opt; write_json {net}'
    subprocess.run([yosys,'-Q','-T','-p',script],check=True,capture_output=True,text=True)
    modules=json.loads(net.read_text())['modules']
    types=[cell['type'] for m in modules.values() for cell in m['cells'].values()]
    assert not any(t.startswith('$mem') or t in ('$mul','$macc') for t in types)
