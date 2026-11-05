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
    # single off-training axis: adding it back reproduces the input
    ev = [{"experiment": "all", "applicable": True, "undone": ["contrast"], "err_after": 0.01}]
    assert attribute(0.5, ev) == {"contrast"}
    ev[0]["err_after"] = 0.2
    assert attribute(0.5, ev) == {"contrast", "geometry"}
    # nothing off-training and a small error: no cause (the old rule blamed geometry here)
    assert attribute(0.007, [{"experiment": "all", "applicable": False}]) == set()
    assert attribute(0.08, [{"experiment": "all", "applicable": False}]) == {"geometry"}
    # interaction: contrast dominates the input, but grid alone still triples the baseline error
    ev = [{"experiment": "all", "applicable": True, "undone": ["grid", "contrast"], "err_after": 0.006},
          {"experiment": "keep_grid", "applicable": True, "err_after": 0.018},
          {"experiment": "keep_contrast", "applicable": True, "err_after": 0.5}]
    assert attribute(0.8, ev) == {"grid", "contrast"}


def test_domain_changes_what_counts_as_off_training():
    from flowguard.diagnose import undo_axis
    broad = {"grid": 128, "contrast_max": 1000.0, "continuous": True}
    a = pdebench(1, np.random.default_rng(1), a_low=0.01)[0]          # contrast 100
    assert undo_axis("contrast", a) is not None and undo_axis("contrast", a, broad) is None
    c = np.exp(np.linspace(np.log(1e-4), 0, 128 * 128)).reshape(128, 128)
    assert undo_axis("continuous", c, broad) is None
    assert np.isclose(undo_axis("contrast", c, broad).max() / undo_axis("contrast", c, broad).min(), 1000.0)


def test_geometry_rule_compares_with_training_shapes():
    base = [{"experiment": "all", "applicable": True, "undone": ["contrast"], "err_after": 0.05}]
    hard = base + [{"experiment": "geometry", "applicable": True, "err_after": 0.04}]
    easy = base + [{"experiment": "geometry", "applicable": True, "err_after": 0.01}]
    # same baseline error: a geometry cause only when training-family shapes with the same values do much better
    assert "geometry" not in attribute(0.06, hard, id_err=0.007)
    assert "geometry" in attribute(0.06, easy, id_err=0.007)
