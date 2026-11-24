"""Round trip of the trim_regression error cache through errio."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "vad_bench", "trim_regression", "scripts"))
from errio import err_from_json, err_to_json  # noqa: E402


def test_roundtrip():
    r = dict(units={("train", "mix|a", 3): [2, 10], ("clip", 0): [0, 5]},
             sdi={"S": 1, "D": 0, "I": 1},
             events=[("S", 1, 2), ("D", 4, None), ("I", None, 7)],
             nref=15, nhyp=14, nseg=2)
    assert err_from_json(json.loads(json.dumps(err_to_json(r)))) == r
