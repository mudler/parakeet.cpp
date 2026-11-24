"""Round trip of the noise_dive e_set clips through the shipped JSON codec."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "vad_bench", "noise_dive"))
from clipio import ClipEncoder, dump_clips, load_clips  # noqa: E402


def _clips():
    rng = np.random.default_rng(0)
    return [
        {
            "name": "speech_00", "cond": "clean", "kind": "speech", "spans": [[0.5, 2.3]],
            "z": {"redux": rng.standard_normal(7).astype(np.float32),
                  "ultra": rng.standard_normal(7).astype(np.float32)},
            "sil": rng.random(7).astype(np.float32),
            # e_build.py computes e10 in float64
            "e10": 10 * np.log10(rng.random(7) + 1e-12),
            "snr": np.float64(12.345678901234567),
            "n": np.int64(7),
            "flag": np.bool_(True),
        },
        {
            "name": "noise_00", "cond": "nz-white20", "kind": "noise", "spans": [], "stretch": [0.0, 1.0],
            "z": {"redux": np.array([-1.0, np.inf], dtype=np.float32), "ultra": np.zeros(0, dtype=np.float32)},
            "sil": np.array([0.0, 0.1], dtype=np.float32),
            "e10": np.array([-40.0, -45.0], dtype=np.float64),
        },
    ]


def _roundtrip(clips):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e_set.json")
        dump_clips(clips, p)
        return load_clips(p)


def test_arrays_keep_dtype_and_exact_values():
    clips = _clips()
    out = _roundtrip(clips)
    assert len(out) == len(clips)
    for a, b in zip(clips, out):
        assert a["name"] == b["name"] and a["spans"] == b["spans"]
        for m in a["z"]:
            assert b["z"][m].dtype == a["z"][m].dtype == np.float32
            np.testing.assert_array_equal(a["z"][m], b["z"][m])
        for k in ("sil", "e10"):
            assert b[k].dtype == a[k].dtype
            np.testing.assert_array_equal(a[k], b[k])
    assert out[0]["e10"].dtype == np.float64


def test_float64_is_lossless():
    x = np.random.default_rng(1).standard_normal(1000) * 1e-9 + 1 / 3
    out = _roundtrip([{"e10": x}])
    assert out[0]["e10"].dtype == np.float64
    assert out[0]["e10"].tobytes() == x.tobytes()


def test_numpy_scalars_become_plain_numbers():
    out = _roundtrip(_clips())
    assert out[0]["snr"] == 12.345678901234567
    assert out[0]["n"] == 7 and out[0]["flag"] is True


def test_encoder_rejects_unknown_types():
    try:
        json.dumps({"x": object()}, cls=ClipEncoder)
    except TypeError:
        return
    raise AssertionError("expected TypeError")
