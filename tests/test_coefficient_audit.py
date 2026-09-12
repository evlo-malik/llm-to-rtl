import numpy as np
import pytest
from compiler.emit import emit_linear
from scripts.audit_coefficients import audit_linear, audit_embedding
from compiler.model_rtl import emit_embedding


def test_reconstruct_and_reject_changed_arithmetic(tmp_path):
    w = np.array([[7, -9, 0], [-128, 127, 1]], dtype=np.int64)
    b = np.array([3, -7])
    path = tmp_path / "linear.sv"
    emit_linear(path, "linear", w, b)
    assert audit_linear(path, dict(w=w, b=b))["coefficients"] == 6
    text = path.read_text()
    path.write_text(text.replace("x0 <<< 3", "x0 <<< 4", 1))
    with pytest.raises(ValueError, match="differs"):
        audit_linear(path, dict(w=w, b=b))


def test_reject_missing_rows(tmp_path):
    path = tmp_path / "linear.sv"
    w = np.array([[1, 2]], dtype=np.int64)
    emit_linear(path, "linear", w)
    path.write_text(
        "\n".join(x for x in path.read_text().splitlines() if " a0 =" not in x)
    )
    with pytest.raises(ValueError, match="missing"):
        audit_linear(path, dict(w=w))


def test_embedding_chunks_and_signed_values(tmp_path):
    q = dict(
        D=72,
        V=3,
        T=2,
        residual_bits=32,
        tok_emb=np.arange(216, dtype=np.int64).reshape(3, 72) - 150,
        pos_emb=np.zeros((2, 72), dtype=np.int64),
    )
    path = tmp_path / "embedding.sv"
    emit_embedding(path, q)
    assert audit_embedding(path, q)["token_rows"] == 3
    q["tok_emb"][0, 0] += 1
    with pytest.raises(ValueError, match="embedding differs"):
        audit_embedding(path, q)
