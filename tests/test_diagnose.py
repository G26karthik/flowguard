import numpy as np

from flowguard.diagnose import _d4, _d4_inv, _undo_contrast, _undo_continuous, _undo_grid, attribute
from flowguard.fields import pdebench


def test_d4_inverse_roundtrip():
    x = np.arange(16.0).reshape(4, 4)
    for k in range(8):
        np.testing.assert_array_equal(_d4_inv(_d4(x, k), k), x)
    assert len({_d4(x, k).tobytes() for k in range(8)}) == 8


def test_undo_edits_reach_training_axes_and_keep_geometry():
    a = pdebench(1, np.random.default_rng(0), a_low=0.001)[0]
    b = _undo_contrast(a)
    assert np.isclose(b.max() / b.min(), 10.0)
    assert np.array_equal(a > 0.5, b > 0.5)  # same geometry
    assert _undo_contrast(pdebench(1, np.random.default_rng(0))[0]) is None
    assert _undo_grid(np.ones((256, 256))).shape == (128, 128) and _undo_grid(np.ones((128, 128))) is None
    c = np.exp(np.linspace(np.log(0.1), 0, 128 * 128)).reshape(128, 128)
    assert set(np.unique(_undo_continuous(c))) == {c.min(), c.max()}


def test_attribute_rule():
    ev = [{"experiment": "contrast", "applicable": True, "err_after": 0.01},
          {"experiment": "grid", "applicable": False},
          {"experiment": "all", "applicable": True, "err_after": 0.01}]
    assert attribute(0.5, ev) == {"contrast"}
    ev[-1]["err_after"] = 0.2
    assert attribute(0.5, ev) == {"contrast", "geometry"}
